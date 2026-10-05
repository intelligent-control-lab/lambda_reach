// Deterministic test driver for the ORIGINAL deployment headers, not a policy port.
// Hardware transport/keyboard/LED IO are replaced by explicit fixture inputs.
// See verify_spark_replacement.py for the recorded source hashes and test scope.
#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <mujoco/mujoco.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "reference_bridge.h"

static std::vector<float> command(3, 0);
namespace isaaclab::mdp {
REGISTER_OBSERVATION(keyboard_velocity_commands) { return command; }
}

template<class T> void read_values(T* values, size_t n) {
    if (fread(values, sizeof(T), n, stdin) != n) throw std::runtime_error("short input");
}
template<class T> void write_values(const T* values, size_t n) {
    if (fwrite(values, sizeof(T), n, stdout) != n) throw std::runtime_error("short output");
}

struct Robot : isaaclab::Articulation {
    mjModel* model;
    mjData* state;
    int quat, gyro, position, velocity;
    Robot(mjModel* m, mjData* d) : model(m), state(d) {
        auto sensor = [&](const char* name) {
            int id = mj_name2id(m, mjOBJ_SENSOR, name);
            if (id < 0) throw std::runtime_error(std::string("missing sensor ") + name);
            return m->sensor_adr[id];
        };
        quat=sensor("imu_quat"); gyro=sensor("imu_gyro");
        position=sensor("frame_pos"); velocity=sensor("frame_vel");
    }
    void update() override {
        // Same float32 DDS payload and BaseArticulation conversion as the source.
        data.root_quat_w = Eigen::Quaternionf(state->sensordata[quat],
            state->sensordata[quat+1],state->sensordata[quat+2],state->sensordata[quat+3]);
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        for (int i=0; i<3; ++i) data.root_ang_vel_b[i]=state->sensordata[gyro+i];
        for (int i=0; i<29; ++i) {
            data.joint_pos[i]=state->sensordata[i];
            data.joint_vel[i]=state->sensordata[i+29];
        }
    }
};

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("config actor xml value-or-dash dt required");
    spdlog::set_level(spdlog::level::off);
    char error[2048]{};
    mjModel* m=mj_loadXML(argv[3], nullptr, error, sizeof(error));
    if (!m) throw std::runtime_error(error);
    m->opt.timestep=std::stod(argv[5]);
    mjData* d=mj_makeData(m);
    mj_forward(m,d);
    auto robot=std::make_shared<Robot>(m,d);
    auto cfg=YAML::LoadFile(argv[1]);
    // Command values are replayed directly; keyboard key-hold timing is outside scope.
    isaaclab::ManagerBasedRLEnv env(cfg,robot);
    isaaclab::OrtRunner actor(argv[2]);
    std::unique_ptr<isaaclab::SafetyMonitor> monitor;
    if (std::string(argv[4])!="-") {
        isaaclab::SafetyMonitorConfig sc;
        sc.model_path=argv[4]; sc.safety_signal_type="avoid";
        sc.safety_signal_phi_max=M_PI/4;
        monitor=std::make_unique<isaaclab::SafetyMonitor>(sc);
    }
    int32_t sizes[]={static_cast<int32_t>(m->nq),static_cast<int32_t>(m->nv),static_cast<int32_t>(m->nsensordata),
        static_cast<int32_t>(env.observation_manager->compute().at("obs").size())};
    write_values(sizes,4); fflush(stdout);
    auto kp=cfg["stiffness"].as<std::vector<float>>();
    auto kd=cfg["damping"].as<std::vector<float>>();
    std::vector<float> target(29);
    double contact_peaks[2]={0,0};
    int ball_id=mj_name2id(m,mjOBJ_BODY,"moving_ball");
    int torso_id=mj_name2id(m,mjOBJ_BODY,"torso_link");
    int op;
    while ((op=getchar())!=EOF && op!='Q') {
        if (op=='R' || op=='S') {
            if (op=='R') mj_resetData(m,d);
            read_values(d->qpos,m->nq); read_values(d->qvel,m->nv);
            mj_forward(m,d); robot->update();
            if (op=='R') {
                // Defer history reset until the command and MoCap sample arrive.
                env.action_manager->reset();
                if (monitor) monitor->last_avoid_collision_time_s_=-INFINITY;
            }
        } else if (op=='F') {
            write_values(d->qpos,m->nq); write_values(d->qvel,m->nv);
            write_values(d->sensordata,m->nsensordata);
            write_values(&d->time,1); write_values(contact_peaks,2);
        } else if (op=='B') {
            double ball[6]; read_values(ball,6);
            int joint=mj_name2id(m,mjOBJ_JOINT,"moving_ball_joint");
            if(joint<0) throw std::runtime_error("no ball in reference model");
            int q=m->jnt_qposadr[joint],v=m->jnt_dofadr[joint];
            for(int i=0;i<3;++i) {d->qpos[q+i]=ball[i];d->qvel[v+i]=ball[i+3];}
            for(int i=0;i<4;++i) d->qpos[q+3+i]=(i==0);
            for(int i=3;i<6;++i) d->qvel[v+i]=0;
            for(int i=0;i<6;++i) d->qacc_warmstart[v+i]=0;
            contact_peaks[0]=0;
            mj_forward(m,d);
        } else if (op=='O' || op=='I') {
            read_values(command.data(),3);
            float relative[6]; read_values(relative,6);
            auto& mocap=isaaclab::MocapClient::instance();
            for (int i=0;i<3;++i) {
                mocap.latest_.rel_position_world[i]=relative[i];
                mocap.latest_.rel_velocity_world[i]=relative[i+3];
            }
            robot->update();
            if (op=='I') env.observation_manager->reset();
            auto obs=env.observation_manager->compute();
            auto action=actor.act(obs);
            env.action_manager->process_action(action);
            target=env.action_manager->processed_actions();
            write_values(obs.at("obs").data(),sizes[3]);
            write_values(action.data(),29); write_values(target.data(),29);
            if (monitor) {
                isaaclab::reference_sim_time=d->time;
                auto prediction=monitor->record(obs,robot->data,{},0);
                float out[]={prediction.value,prediction.signal};
                write_values(out,2);
            }
        } else if (op=='P' || op=='C') {
            read_values(target.data(),29);
            int32_t substeps; read_values(&substeps,1);
            contact_peaks[0]=contact_peaks[1]=0;
            for (int n=0;n<substeps;++n) {
                // Diagnostic only: normalize sensor/control timing to SPARK.
                if(op=='C') mj_forward(m,d);
                // Exact torque expression extracted from the original SDK bridge.
                reference_control(d,target.data(),kp.data(),kd.data());
                mj_step(m,d);
                for(int i=0;i<d->ncon;++i) {
                    auto& c=d->contact[i];
                    int b1=m->geom_bodyid[c.geom1], b2=m->geom_bodyid[c.geom2];
                    double force[6]; mj_contactForce(m,d,i,force);
                    double norm=std::sqrt(force[0]*force[0]+force[1]*force[1]+force[2]*force[2]);
                    if(b1==ball_id || b2==ball_id) contact_peaks[0]=std::max(contact_peaks[0],norm);
                    if(b1==torso_id || b2==torso_id) contact_peaks[1]=std::max(contact_peaks[1],norm);
                }
            }
            if(op=='C') mj_forward(m,d);
        } else throw std::runtime_error("unknown operation");
        fflush(stdout);
    }
    mj_deleteData(d); mj_deleteModel(m);
    return 0;
  } catch (const std::exception& exc) {
    fprintf(stderr,"native reference: %s\n",exc.what());
    return 1;
  }
}
