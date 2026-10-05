import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from model import features, make_model

def train(data, output):
    X = features(data)
    if 'Exited' not in data or set(data.Exited.dropna().unique()) != {0,1} or data.Exited.isna().any():
        raise ValueError('Exited debe contener ambas clases 0 y 1, sin nulos')
    X_train,X_test,y_train,y_test = train_test_split(X,data.Exited,test_size=0.2,stratify=data.Exited,random_state=42)
    model = make_model().fit(X_train,y_train)
    baseline = DummyClassifier(strategy='prior').fit(X_train,y_train)
    def metrics(clf):
        p = clf.predict(X_test); prob = clf.predict_proba(X_test)[:,list(clf.classes_).index(1)]
        return dict(accuracy=accuracy_score(y_test,p),precision=precision_score(y_test,p,zero_division=0),recall=recall_score(y_test,p,zero_division=0),f1=f1_score(y_test,p,zero_division=0),roc_auc=roc_auc_score(y_test,prob),confusion_matrix=confusion_matrix(y_test,p,labels=[0,1]).tolist())
    result = {'split':'80/20 estratificado','seed':42,'train_rows':len(X_train),'test_rows':len(X_test),'threshold':0.5,'baseline':metrics(baseline),'random_forest':metrics(model)}
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output/'pipeline.joblib')
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return model,result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    args=parser.parse_args()
    _,result=train(pd.read_csv(args.csv),args.output)
    print(json.dumps(result,indent=2))
