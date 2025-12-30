import subprocess
from pathlib import Path

from src.data.preprocessing import load_data, preprocess_data , save_preprocess
from src.models.train import train_model
from src.models.evaluate import evaluate



## PATH

Raw_Datapath= Path("data/raw/creditcard.csv")
Preprocess_Datapath=Path("data/processed/preprocess_creditcard.csv")


## Preprocess Data
df_raw= load_data(Raw_Datapath)

df_processed=preprocess_data(df_raw)

print("Saving Processed Data......")
save_preprocess(df_processed,Preprocess_Datapath)


## Tracking process with DVC

print(" Tracking processed data with DVC...")

try: 
    subprocess.run(["dvc", "add" ,str(Preprocess_Datapath)], check =True)
    subprocess.run(["git","add",str(Preprocess_Datapath)+".dvc"],check=True)
    subprocess.run(["git","commit","-m","Add processed data via DVC"],check=True)

except subprocess.CalledProcessError:
    print("DVC tracking skipped or already exists.")


## Train Model

print("Training Model.....")

clf,X_test,y_test=train_model(df_processed)


## Evaluate Model

print("Evaluating Model......")
evaluate()

print("Mlflow tracking done in train model step . Run `Mlflow ui` to visualize the experiments.")

print ("Workflow completed")

