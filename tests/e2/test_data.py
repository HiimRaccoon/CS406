from types import SimpleNamespace
from pathlib import Path
import copy
import numpy as np
from PIL import Image
import pytest
import torch
from src.preprocessing.geoguessr_v1 import Preprocessor,Photometric

def fixture_dataset(tmp_path,wh,recipe):
    from src.e2.data import E2Dataset
    label=tmp_path/'Aland';label.mkdir(exist_ok=True)
    yy,xx=np.indices((wh[1],wh[0]))
    image=Image.fromarray(np.stack([xx%256,yy%256,(xx+yy)%256],-1).astype('uint8'))
    path=label/'a.png';image.save(path)
    record=SimpleNamespace(relative_path='Aland/a.png',label_id=0,width=wh[0],height=wh[1])
    manifest=SimpleNamespace(records=lambda split:(record,))
    return E2Dataset(manifest,tmp_path,'train',1001,recipe['preprocessing']['augmentation']),image

@pytest.mark.parametrize('wh',[(1536,662),(1030,703)])
def test_both_layouts_global_and_locals_match_frozen_preprocessor(tmp_path,wh,recipe):
    d,image=fixture_dataset(tmp_path,wh,recipe)
    views,y,path=d[0]
    assert set(views)=={'global','local'}
    assert views['global'].shape==(3,384,896) and views['local'].shape==(4,3,320,320)
    assert y==0 and path=='Aland/a.png'
    assert all(torch.equal(v,d[0][0][k]) for k,v in views.items())

def test_one_jitter_pair_is_shared_by_all_views(tmp_path,recipe):
    from src.e2.data import derived_seed
    d,image=fixture_dataset(tmp_path,(1030,703),recipe)
    generator=torch.Generator().manual_seed(derived_seed(1001,0,'Aland/a.png','jitter'))
    factors=Photometric(*[.9+.2*torch.rand((),generator=generator).item() for _ in range(2)])
    processor=Preprocessor('e2');raw,_,layout=processor.processed_raw(image,factors)
    got=d[0][0]
    assert torch.equal(got['global'],processor.normalize(processor.letterbox(raw)))
    for i,(x,y,w,h) in enumerate(layout['local_crops_xywh']):
        expected=processor.normalize(processor.resize(raw[:,y:y+h,x:x+w],(320,320)))
        assert torch.equal(got['local'][i],expected)
    assert not torch.equal(got['local'],d[(1,0)][0]['local'])

@pytest.mark.parametrize('batch',[1,2])
def test_collation_and_transfer_preserve_original_images(batch):
    from src.e2.data import move_views
    views={'global':torch.randn(batch,3,384,896),'local':torch.randn(batch,4,3,320,320)}
    moved=move_views(views,device=torch.device('cpu'),batch_size=batch)
    assert set(moved)==set(views)
    assert all(torch.equal(moved[k],v) for k,v in views.items())
    for bad in [{'global':views['global']},{**views,'extra':views['global']}, {**views,'local':views['local'].double()}, {**views,'local':views['local'][:,:3]}]:
        with pytest.raises(ValueError):move_views(bad,device=torch.device('cpu'),batch_size=batch)

def test_test_rejected_before_open_and_metadata_is_fixed(monkeypatch,tmp_path,recipe):
    from src.e2.data import E2Dataset,FixedManifest
    m=FixedManifest()
    assert len(m.records('train'))==34997 and len(m.records('val'))==7497
    monkeypatch.setattr(Image,'open',lambda *a,**kw:pytest.fail('must reject test before open'))
    with pytest.raises(ValueError):E2Dataset(m,tmp_path,'test',1001,{})
    with pytest.raises(ValueError):m.verify_images(tmp_path,('train','test'))
