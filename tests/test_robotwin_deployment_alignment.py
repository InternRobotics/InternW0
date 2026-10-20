import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from w0.datasets.robotwin_posttrain import RECIPE, RobotwinPosttrainProcessor, camera_canvas
from experiments.robotwin.eval_config import checkpoint_tag, resolve_training_config
from experiments.robotwin.w0_policy import deploy_policy as deploy
from experiments.robotwin.w0_policy.fixed_clock import simulation_steps, take_fixed_action
from experiments.robotwin import eval_robotwin_single as single
from experiments.robotwin import run_robotwin_manager as manager


def stats_file(tmp_path):
    stats = dict(recipe=RECIPE, metadata_sha256='test', split_seed=42, val_set_proportion=.01)
    for kind in ('action', 'proprio'):
        stats[kind] = dict(q01=[-2.] * 14, q99=[2.] * 14, minmax_bounds={})
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(stats))
    return path


class FakeModel(torch.nn.Module):
    device = torch.device('cpu')
    torch_dtype = torch.float32
    proprio_dim = 37
    action_horizon = 64
    action_chunk_size = 16

    def __init__(self, stats):
        super().__init__()
        self.calls = []
        for name in ('mot', 'proprio_encoder', 'action_obs_visual_proj', 'chunk_obs_query_encoder'):
            setattr(self, name, torch.nn.Linear(1, 1))
        self.stats = stats
        self._inference_state = None

    def encode_prompt(self, prompt):
        return torch.ones(1, 128, 4096), torch.arange(128)[None] < 7

    def load_checkpoint(self, path):
        result = {name: getattr(self, name).state_dict() for name in
                  ('mot', 'proprio_encoder', 'action_obs_visual_proj', 'chunk_obs_query_encoder')}
        result['posttrain_statistics'] = self.stats
        result['training_recipe'] = {'data_offset_config': {'train': {'domains': dict(
            target_fps=12.5, metadata_sha256='test', split_seed=42, val_set_proportion=.01)}}}
        return result

    def infer_action(self, phase='video', **kwargs):
        self.calls.append((phase, kwargs))
        if phase == 'video':
            # Exercise the real model's horizon contract.
            from w0.models.wan22.w0_chunk_base import W0ChunkBase
            W0ChunkBase._validate_runtime_action_horizon(self, kwargs['action_horizon'])
            assert kwargs['context'].shape == (128, 4096)
            assert kwargs['context_mask'].all()
            assert not kwargs['context'][7:].any()
            self._inference_state = {'next_chunk_index': 0}
            return {}
        idx = self._inference_state['next_chunk_index']
        self._inference_state['next_chunk_index'] += 1
        return dict(action_chunk=torch.zeros(16, 37), chunk_index=idx)


def make_policy(tmp_path, chunks=1):
    path = stats_file(tmp_path)
    stats = json.loads(path.read_text())
    model = FakeModel(stats)
    processor_cfg = OmegaConf.create(dict(_target_='w0.datasets.robotwin_posttrain.RobotwinPosttrainProcessor', stats_path=str(path)))
    def instantiate(cfg, **kwargs):
        if cfg.get('_target_') == processor_cfg._target_:
            return RobotwinPosttrainProcessor(cfg.stats_path)
        assert cfg.tokenizer_max_len == 128
        return model
    with patch.object(deploy, 'instantiate', side_effect=instantiate):
        policy = deploy.WorldActionRobotWinPolicy(
            model_cfg=OmegaConf.create({}), processor_cfg=processor_cfg,
            checkpoint_path='fake.pt', dataset_stats_path=path, device='cpu',
            model_dtype=torch.float32, action_horizon=64, chunks_per_video_prefill=chunks,
            num_inference_steps=10, sigma_shift=None, seed=42, text_cfg_scale=1.,
            negative_prompt='', rand_device='cpu', tiled=False, timing_enabled=False,
            control_mode='fixed_period', target_fps=12.5,
        )
    return policy, model


def observation():
    rng = np.random.default_rng(4)
    return dict(observation={name: dict(rgb=rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
                             for name in ('head_camera', 'left_camera', 'right_camera')},
                joint_action={'vector': np.linspace(-1, 1, 14)})


@pytest.mark.parametrize('chunks', [1, 4])
def test_policy_warmup_full_horizon_text_and_replanning(tmp_path, chunks):
    policy, model = make_policy(tmp_path, chunks)
    assert model.calls[0][0] == 'video'
    assert model.calls[0][1]['action_horizon'] == 64
    obs = observation()
    model.calls.clear()
    for i in range(chunks + 1):
        policy._fill_action_queue(obs, 'move object')
        assert len(policy.pending_actions) == 16
        policy.pending_actions.clear()
    video_calls = [kwargs for phase, kwargs in model.calls if phase == 'video']
    assert len(video_calls) == 2
    assert all(c['action_horizon'] == 64 for c in video_calls)
    policy.reset()
    assert policy._episode_prefilled is False and model._inference_state is None


def test_shared_canvas_float32_and_native_roundtrip(tmp_path):
    policy, _ = make_policy(tmp_path)
    obs = observation()
    canvas = policy._build_robotwin_image_tensor(obs)
    images = [torch.from_numpy(obs['observation'][k]['rgb']).permute(2, 0, 1)[None]
              for k in ('head_camera', 'left_camera', 'right_camera')]
    assert torch.equal(canvas, camera_canvas(images))
    assert canvas.dtype == torch.float32 and canvas.shape == (1, 3, 384, 320)
    state = obs['joint_action']['vector'].astype(np.float32)
    assert torch.equal(policy._normalize_state(state), policy.processor.pack(torch.from_numpy(state)[None], 'proprio')[0])
    packed, _ = policy.processor.pack(torch.from_numpy(state).repeat(16, 1), 'action')
    np.testing.assert_allclose(policy._denormalize_action(packed)[0], np.tile(state, (16, 1)), atol=3e-7)


def test_cached_text_uses_exact_training_representation(tmp_path):
    policy, model = make_policy(tmp_path)
    policy.text_cache_dir = tmp_path
    prompt = 'cached instruction'
    context = torch.rand(128, 4096)
    mask = torch.arange(128) < 11
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    torch.save({'context': context, 'mask': mask}, tmp_path / f'{digest}.t5_len128.wan22ti2v5b.pt')
    model.encode_prompt = lambda _: pytest.fail('Cache should avoid online text encoding')
    kwargs = policy._prompt_kwargs(prompt)
    expected = context.clone(); expected[~mask] = 0
    assert torch.equal(kwargs['context'], expected) and kwargs['context_mask'].all()


def test_checkpoint_rate_and_partial_weights_rejected(tmp_path):
    _, model = make_policy(tmp_path)
    payload = model.load_checkpoint('fake')
    with pytest.raises(ValueError, match='target_fps'):
        deploy.validate_posttrain_checkpoint(model, payload, 25)
    payload['mot'].pop('bias')
    with pytest.raises(ValueError, match='Incomplete'):
        deploy.validate_posttrain_checkpoint(model, payload, 12.5)


def fake_env(success_step=None):
    positions = dict(left=np.zeros(6), right=np.zeros(6))
    grippers = dict(left=0., right=0.)
    calls = []
    def set_arm(point, velocity, side):
        positions[side] = point.copy(); calls.append((side, point.copy(), velocity.copy()))
    robot = SimpleNamespace(
        get_left_arm_jointState=lambda: positions['left'].tolist() + [grippers['left']],
        get_right_arm_jointState=lambda: positions['right'].tolist() + [grippers['right']],
        set_arm_joints=set_arm, set_gripper=lambda x, side: grippers.update({side: x}))
    env = SimpleNamespace(robot=robot, take_action_cnt=0, step_lim=100, eval_success=False,
                          eval_video_path=None, render_freq=0, simulation_ticks=0)
    def step(): env.simulation_ticks += 1
    env.scene = SimpleNamespace(get_timestep=lambda: .004, step=step)
    env._update_render = lambda: None
    env.get_obs = lambda: None
    env.check_success = lambda: success_step is not None and env.simulation_ticks == success_step
    return env, positions, grippers, calls


@pytest.mark.parametrize('fps,steps', [(12.5, 20), (25., 10)])
def test_fixed_clock_exact_duration_endpoint_and_termination(fps, steps):
    env, pos, grip, calls = fake_env()
    action = np.linspace(0, 1, 14)
    take_fixed_action(env, action, fps)
    assert env.simulation_ticks == steps and env.take_action_cnt == 1
    np.testing.assert_allclose(pos['left'], action[:6])
    np.testing.assert_allclose(pos['right'], action[7:13])
    assert grip['left'] == action[6] and grip['right'] == action[13]
    np.testing.assert_allclose(calls[0][1], action[:6] / steps)
    np.testing.assert_allclose(calls[0][2], action[:6] * fps)
    env, _, _, _ = fake_env(success_step=3)
    take_fixed_action(env, action, fps)
    take_fixed_action(env, action, fps)
    assert env.simulation_ticks == 3 and env.eval_success and env.take_action_cnt == 1


def test_fixed_clock_rejects_inexact_period_and_bad_action():
    with pytest.raises(ValueError, match='integer'):
        simulation_steps(15., .004)
    env, _, _, _ = fake_env()
    with pytest.raises(ValueError, match='14 finite'):
        take_fixed_action(env, [float('nan')] * 14, 12.5)
    assert env.take_action_cnt == 0 and env.simulation_ticks == 0


def test_policy_control_report_and_equal_simulation_budget(tmp_path):
    policy, _ = make_policy(tmp_path)
    policy.target_fps = 25
    policy.step_limit_reference_fps = 12.5
    policy.controller_report_path = tmp_path / 'controller.json'
    env, _, _, _ = fake_env()
    env.get_instruction = lambda: 'move object'
    policy.step(env, observation())
    assert env.step_lim == 200  # 100/12.5 == 200/25 seconds.
    assert env.simulation_ticks == 10
    report = json.loads(policy.controller_report_path.read_text())
    assert report['max_simulation_seconds'] == 8
    assert report['simulation_steps_per_action'] == 10
    policy.step(env, None)
    assert env.step_lim == 200 and env.simulation_ticks == 20


def config_fixture(tmp_path, stride=2):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='eval_robotwin')
        trained = compose(config_name='posttrain_robotwin')
    with open_dict(trained.data.train):
        trained.data.train.sampling_stride = stride
    trained.posttrain_stats = str(stats_file(tmp_path))
    OmegaConf.resolve(trained)
    path = tmp_path / 'experiment.yaml'
    OmegaConf.save(trained, path)
    cfg.EVALUATION.training_config = str(path)
    return cfg, trained


def test_training_config_is_authoritative_and_worker_roundtrip(tmp_path):
    cfg, trained = config_fixture(tmp_path)
    cfg.model.max_action_offset = 15
    cfg = resolve_training_config(cfg, tmp_path / '2500.pt')
    assert cfg.model.max_action_offset == 0
    assert cfg.EVALUATION.target_fps == 25 and cfg.EVALUATION.sampling_stride == 2
    assert cfg.EVALUATION.dataset_stats_path == trained.posttrain_stats
    assert OmegaConf.to_container(cfg.model, resolve=True) == OmegaConf.to_container(trained.model, resolve=True)
    snapshot = tmp_path / 'resolved.yaml'; OmegaConf.save(cfg, snapshot)
    captured = {}
    def policy(**kwargs): captured.update(kwargs); return 'policy'
    with patch.object(deploy, 'WorldActionRobotWinPolicy', side_effect=policy):
        assert deploy.get_model({'resolved_config_path': str(snapshot), 'ckpt_setting': '2500.pt', 'device': 'cpu'}) == 'policy'
    assert captured['model_cfg'].max_action_offset == 0
    assert captured['target_fps'] == 25 and captured['control_mode'] == 'fixed_period'
    assert captured['chunks_per_video_prefill'] == 4 and captured['action_horizon'] == 64


def test_missing_training_config_and_explicit_missing_stats_fail(tmp_path):
    cfg, _ = config_fixture(tmp_path)
    cfg.EVALUATION.training_config = str(tmp_path / 'missing.yaml')
    with pytest.raises(FileNotFoundError, match='original resolved'):
        resolve_training_config(cfg, tmp_path / '2500.pt')
    cfg.EVALUATION.dataset_stats_path = str(tmp_path / 'missing.json')
    (tmp_path / 'dataset_stats.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='Explicit'):
        single._resolve_dataset_stats_path(cfg, tmp_path / '2500.pt')


def test_checkpoint_step_tags_and_external_task_root(tmp_path):
    a = tmp_path / 'runs/job/training/checkpoints/weights/2500.pt'
    b = a.with_name('5000.pt')
    assert checkpoint_tag(a) != checkpoint_tag(b)
    assert single._resolve_ckpt_tag(a) == manager._resolve_ckpt_tag(a)
    (tmp_path / 'task_config').mkdir()
    (tmp_path / 'task_config/_eval_step_limit.yml').write_text('task_a: 100\ntask_b: 200\n')
    assert manager._load_all_tasks(tmp_path) == ['task_a', 'task_b']


def test_single_entry_forwards_resolved_config_and_honors_output(tmp_path, monkeypatch):
    cfg, _ = config_fixture(tmp_path)
    checkpoint = tmp_path / '2500.pt'; checkpoint.touch()
    robotwin = tmp_path / 'RoboTwin'; (robotwin / 'policy').mkdir(parents=True)
    cfg.ckpt = str(checkpoint)
    cfg.EVALUATION.task_name = 'test_task'
    cfg.EVALUATION.robotwin_root = str(robotwin)
    cfg.EVALUATION.output_dir = str(tmp_path / 'requested_results')
    captured = []
    class Process:
        stdout = []
        def wait(self): return 0
    def popen(cmd, **kwargs):
        assert kwargs['cwd'] == str(robotwin)
        pair = cmd[cmd.index('--overrides') + 1:]
        opts = {k: v for k, v in zip(pair[::2], pair[1::2])}
        import ast
        forwarded = OmegaConf.load(ast.literal_eval(opts['--resolved_config_path']))
        assert forwarded.model.max_action_offset == 0
        assert forwarded.EVALUATION.target_fps == 25
        assert forwarded.EVALUATION.control_mode == 'fixed_period'
        captured.append(ast.literal_eval(opts['--eval_output_dir']))
        return Process()
    monkeypatch.setattr(single.subprocess, 'Popen', popen)
    single.main.__wrapped__(cfg)
    assert captured == [str(tmp_path / 'requested_results' / checkpoint_tag(checkpoint) / 'test_task')]


def test_manager_forwards_posttrain_defaults_without_cli_overrides(tmp_path, monkeypatch):
    cfg, _ = config_fixture(tmp_path)
    checkpoint = tmp_path / '2500.pt'; checkpoint.touch()
    robotwin = tmp_path / 'RoboTwin'; robotwin.mkdir()
    cfg.ckpt = str(checkpoint); cfg.EVALUATION.task_name = 'test_task'
    cfg.EVALUATION.robotwin_root = str(robotwin)
    cfg.EVALUATION.output_dir = str(tmp_path / 'results')
    cfg.MULTIRUN.num_gpus = 1
    seen = []
    class Process:
        def poll(self): return 0
    def popen(cmd, **kwargs):
        directory = Path(cmd[cmd.index('--config-path') + 1])
        name = cmd[cmd.index('--config-name') + 1]
        child = OmegaConf.load(directory / (name + '.yaml'))
        assert child.posttrain_required and child.model.max_action_offset == 0
        assert child.EVALUATION.target_fps == 25 and child.EVALUATION.chunks_per_video_prefill == 4
        assert kwargs['start_new_session']
        phase = 'clean' if child.EVALUATION.task_config == 'demo_clean' else 'random'
        result_dir = tmp_path / 'results' / checkpoint_tag(checkpoint) / 'test_task'
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f'_result_{phase}.txt').write_text('success\n0.5\n')
        seen.append(phase)
        return Process()
    monkeypatch.setattr(manager.subprocess, 'Popen', popen)
    manager.main.__wrapped__(cfg)
    assert seen == ['clean', 'random']
    report = json.loads((tmp_path / 'results' / checkpoint_tag(checkpoint) / 'summary.json').read_text())
    assert report['overall']['clean_mean_success_rate'] == .5
