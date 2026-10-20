import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from w0.datasets.motus_posttrain import (
    DOMAIN, RECIPE, MotusPosttrainDataset, MotusPosttrainProcessor, validation_episode,
)
from test_robotwin_posttrain import tiny_model


def make_clean_dataset(tmp_path, length=11, training=True, **temporal):
    task = 'test_task'
    ep = next(i for i in range(10000) if validation_episode(task, i) != training)
    base = tmp_path / task
    for directory in ('qpos', 'metas', 'umt5_wan', 'videos'):
        (base/directory).mkdir(parents=True)
    qpos = torch.arange(length)[:,None].expand(-1,14).float()/length
    torch.save(qpos,base/f'qpos/{ep}.pt')
    (base/f'metas/{ep}.txt').write_text('first\nsecond\n')
    torch.save([torch.ones(8,4096),torch.full((9,4096),2.)],base/f'umt5_wan/{ep}.pt')
    manifest = json.dumps(dict(episodes=[dict(task_name=task,episode_id=ep,source_length=length)]))
    (tmp_path/'dataset_manifest.json').write_text(manifest)
    bounds = dict(q01=[-2.]*14,q99=[2.]*14,minmax_bounds={})
    stats = dict(recipe=RECIPE,action=bounds,proprio=bounds,metadata_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                 split_seed=42,val_set_proportion=.01)
    (tmp_path/'stats.json').write_text(json.dumps(stats))
    processor=MotusPosttrainProcessor(str(tmp_path/'stats.json'))
    dataset=MotusPosttrainDataset(str(tmp_path),processor,
                                  is_training_set=not validation_episode(task,ep), **temporal)
    return dataset, processor, qpos


@pytest.mark.parametrize('offset,horizon', [(15, 96), (0, 96), (0, 64)])
def test_clean_native_next_qpos_slots_short_tail_and_eval(tmp_path, monkeypatch, offset, horizon):
    dataset, processor, qpos = make_clean_dataset(tmp_path, max_action_offset=offset, video_plan_horizon=horizon)
    calls=[]
    def decode(path,positions,fps):
        calls.append((positions.copy(),fps))
        return torch.zeros(len(positions),3,360,320,dtype=torch.uint8)
    monkeypatch.setattr('w0.datasets.motus_posttrain.decode_native_frames',decode)
    assert len(dataset)==9
    sample=dataset[(1,0)]
    torch.testing.assert_close(processor.unpack_action(sample['action'])[0],qpos[3])
    expected,_=processor.pack(qpos[1],'proprio')
    torch.testing.assert_close(sample['proprio'][0],expected)
    assert calls[0][1]==30 and calls[0][0][0]==1
    assert (~sample['action_is_pad']).sum()==4
    assert sample['action_valid_mask'][0].nonzero().flatten().tolist()==RECIPE['slots']
    assert not sample['action'][~sample['action_valid_mask']].any()
    tail=dataset[8]
    assert (~tail['action_is_pad']).sum()==1 and int(tail['action_offset'])==0
    assert tail['image_is_pad'][1:].all()
    assert tail['context'][0,0]==(1 if tail['prompt']=='first' else 2)
    torch.set_num_threads(1)
    model=tiny_model()
    model.max_action_offset = offset
    model.configure_future_video(video_plan_horizon=horizon, video_action_stride=8,
                                 future_training_mode='idm', video_plan_steps=1)
    model.action_expert.domain_configs=[DOMAIN]
    model.action_expert.soft_prompt_domains=[DOMAIN['name']]
    # Keep real data timing but use the tiny model's text/image sizes.
    tail['video']=torch.zeros(3,horizon//8+1,16,16)
    tail['chunk_obs_images']=torch.zeros(4,3,16,16)
    tail['context']=torch.ones(3,32)
    tail['context_mask']=torch.ones(3,dtype=torch.bool)
    from w0.trainer import Wan22Trainer
    batch=Wan22Trainer._to_batched_eval_sample(tail)
    batch['_eval_dataset']=dataset
    model.bind_training_data_recipe(dataset, dataset)
    loss, _ = model.training_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_expert.head.weight.grad is not None
    result=model.evaluate_validation(batch,eval_num_inference_steps=1)
    assert np.isfinite(result['val_loss']) and np.isfinite(result['action_l1'])
