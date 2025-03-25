import pytest
import pandas as pd
from rxnmapper.find_best_head_layer import FindBestHeadLayerCombination

@pytest.fixture
def dummy_rxns():
    return ['Cc1csc2nc(-c3ccno3)nc(Cl)c12.NCc1ccc(F)cc1>>Cc1csc2nc(-c3ccno3)nc(NCc3ccc(F)cc3)c12',
            'CC(O)(c1ccc(N2CCN(S(=O)(=O)c3cccs3)C[C@@H]2CN2C3COCC2C(OCc2ccccc2)C3)cc1)C(F)(F)F.CO.ClCCl>>CC(O)(c1ccc(N2CCN(S(=O)(=O)c3cccs3)C[C@@H]2CN2C3COCC2C(O)C3)cc1)C(F)(F)F',
            'CC(=O)O.CC(=O)O[BH-](OC(C)=O)OC(C)=O.CC=O.Cc1cc(NC(=O)c2ccc3c(c2)CCNC3)nn1Cc1cc(Cl)ccc1OCC(C)C.Cl.ClCCl.[Na+]>>CCN1CCc2cc(C(=O)Nc3cc(C)n(Cc4cc(Cl)ccc4OCC(C)C)n3)ccc2C1.Cl',
            'Br.C1CCOC1.CCOC(C)=O.CN(C)C=O.Clc1ccc2cccc(C3=C(Br)SC4=NCCN43)c2c1.[Cl-].[NH4+]>>O=CC1=C(c2cccc3ccc(Cl)cc23)N2CCN=C2S1',
            'CC(C)I.CCN(CC)CC.CO.Nc1cccc2c1CCO2>>CC(C)Nc1cccc2c1CCO2']

@pytest.fixture
def model_path():
    return '../rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k'

@pytest.fixture
def fbh(dummy_rxns, model_path):
    return FindBestHeadLayerCombination(
        rxns_list=dummy_rxns,
        heads=[0],
        layers=[0],
        model_path=model_path
    )

def test_get_config_dict_to_eval(fbh):
    config = fbh.get_config_dict_to_eval()
    assert isinstance(config, dict)
    assert len(config) == 1
    for key, val in config.items():
        assert isinstance(key, str)
        assert "head" in val and "layer" in val
        assert isinstance(val["head"], int)
        assert isinstance(val["layer"], int)

def test_map_rxns_from_dict_structure(fbh):
    config = fbh.get_config_dict_to_eval()
    model_key = list(config.keys())[0]
    mapped = fbh.map_rxns_from_dict(config[model_key])
    assert isinstance(mapped, list)
    assert len(mapped) == len(fbh.rxns_list)
    for result in mapped:
        assert "mapped_rxn" in result
        assert "confidence" in result

def test_map_rxns_from_models_in_dict_returns_df(fbh):
    fbh.get_config_dict_to_eval()
    df = fbh.map_rxns_from_models_in_dict()
    assert isinstance(df, pd.DataFrame)
    assert "mapped_rxn" in df.columns
    assert "model_id" in df.columns
    assert len(df) == len(fbh.rxns_list)

def test_evaluate_results_outputs_dict(fbh):
    fbh.get_config_dict_to_eval()
    fbh.map_rxns_from_models_in_dict()
    eval_result = fbh.evaluate_results()
    assert isinstance(eval_result, dict)
    for k, v in eval_result.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
