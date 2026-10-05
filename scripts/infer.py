import argparse
from pathlib import Path
import joblib
import pandas as pd
from model import predict

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True,help='Solo modelos generados por ti; joblib ejecuta código al cargar.')
    parser.add_argument('--output',type=Path,default=Path('demo_output/predicciones.csv'))
    args=parser.parse_args()
    result=predict(pd.read_csv(args.csv),joblib.load(args.model))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Predicciones exportadas: {len(result)}')

if __name__=='__main__':
    main()
