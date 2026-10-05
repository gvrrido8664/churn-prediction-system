"""Demo sintética y prueba de consistencia de países entre lotes."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from train import train
from model import predict

def dataset():
    rng=np.random.default_rng(42); n=600
    frame=pd.DataFrame(dict(CreditScore=rng.integers(400,850,n),Age=rng.integers(18,80,n),Tenure=rng.integers(0,10,n),Balance=rng.uniform(0,200000,n),NumOfProducts=rng.integers(1,5,n),HasCrCard=rng.integers(0,2,n),IsActiveMember=rng.integers(0,2,n),EstimatedSalary=rng.uniform(1000,200000,n),Geography=rng.choice(['France','Germany','Spain'],n),Gender=rng.choice(['Female','Male'],n)))
    probability=1/(1+np.exp(-(-2+0.035*(frame.Age-40)+0.8*(frame.Geography=='Germany')+0.9*(1-frame.IsActiveMember))))
    frame['Exited']=(rng.random(n)<probability).astype(int)
    return frame

if __name__=='__main__':
    import json
    output=ROOT/'demo_output'; data=dataset()
    model,metrics=train(data,output)
    metrics['dataset']='SINTÉTICO: prueba de funcionamiento, no desempeño bancario'
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    row=data[data.Geography=='Germany'].iloc[:1]
    a=predict(row,model).Probabilidad_Fuga.iloc[0]
    b=predict(pd.concat([row,data.iloc[:30]]),model).Probabilidad_Fuga.iloc[0]
    assert a==b, 'La composición del lote cambió la predicción'
    other=row.copy(); other['Geography']='Unknown country'
    assert 0<=predict(other,model).Probabilidad_Fuga.iloc[0]<=1
    changed=row.copy(); changed['Exited']=1-changed.Exited
    assert predict(changed,model).Probabilidad_Fuga.iloc[0]==a, 'Fuga de etiqueta'
    data.to_csv(output/'clientes_sinteticos.csv',index=False)
    predict(data,model).to_csv(output/'predicciones.csv',index=False)
    print('Demo y comprobaciones correctas. Métricas sintéticas:',metrics['random_forest'])
