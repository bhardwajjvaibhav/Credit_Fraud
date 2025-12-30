import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RAW_DATAPATH = Path("data/raw/creditcard.csv")
Preprocess_DATAPATH = Path("data/processed/processed_creditcard.csv")

def load_data(path:Path=RAW_DATAPATH) -> pd.DataFrame :

    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found at:{path}")
    
    return pd.read_csv(path)


def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:

    df=df.copy()

    scaler=StandardScaler()

    df["norm_amount"] = scaler.fit_transform(df[["Amount"]])

    df["norm_time"] = scaler.fit_transform(df[["Time"]])

    df=df.drop(columns=["Amount","Time"])

    return df



def save_preprocess(df:pd.DataFrame,path:Path=Preprocess_DATAPATH):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)
    print(f"Processed dataset saved to {path}")


def main():
    
    print("📥 Loading dataset...")
    df = load_data()

    print("⚙️  Preprocessing dataset...")
    processed_df = preprocess_data(df)

    print("💾 Saving processed dataset...")
    save_preprocess(processed_df)


if __name__ == "__main__":
    main()

