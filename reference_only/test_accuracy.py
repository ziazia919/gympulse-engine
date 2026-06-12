import os
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 🗺️ Categorical Map Dictionaries matching your exact data rules
TERRAIN_MAP = {'Low Complexity': 2.0, 'Medium Complexity': 5.0, 'High Complexity': 9.0}
WEATHER_MAP = {'Low Severity': 1.0, 'Medium Severity': 5.0, 'High Severity': 9.0}
SECURITY_MAP = {'Low Risk': 2.0, 'Meduim Risk': 5.0, 'High Risk': 9.0}
URGENCY_MAP = {'Not urgent': 2.0, 'Medium': 5.0, 'Medium urgent': 6.0, 'Most urgent': 9.0, 'Most Urgent': 9.0}
VISIBILITY_MAP = {'Clear Visibility': 10.0, 'Moderate Visibility': 5.0, 'Low Visibility': 2.0}

CAMP_MAP = {
    'Islamabad office': 1.0, 'Islamabad Office': 1.0,
    'Lahore office': 2.0, 'Lahore Office': 2.0,
    'Karachi office': 3.0, 'Karachi Office': 3.0
}

LOCATION_MAP = {
    'Swat-Kalam': 1.0, 'Abbottabad-Pass': 2.0, 'Islamabad-G9': 3.0, 'Azad-Kashmir': 4.0, 
    'Lahore-DHA': 5.0, 'Lahore-Gulberg': 6.0, 'Swat-Mingora': 7.0, 'Islamabad-F7': 8.0, 
    'Islamabad-E11': 9.0, 'Murree-Expressway': 10.0, 'Kashmir-Bagh': 11.0, 'Abbottabad-City': 12.0,
    'Lahore-Cantt': 13.0, 'Swat-Madyan': 14.0, 'Islamabad-I8': 15.0, 'Kashmir-Muzaffarabad': 16.0,
    'Kashmir-Rawalakot': 17.0, 'Lahore-Raiwind': 18.0, 'Islamabad-Swat': 19.0, 'Islamabad-BlueArea': 20.0,
    'Abbottabad-Nathiagali': 21.0, 'Lahore-JoharTown': 22.0, 'Islamabad-G11': 23.0, 'Swat-Khwazakhela': 24.0,
    'Murree-MallRoad': 25.0, 'Lahore-WalledCity': 26.0, 'Abbottabad-H12': 27.0, 'Kashmir-Kotli': 28.0,
    'Abbottabad-Mirpur': 29.0, 'Lahore-ModelTown': 30.0, 'Islamabad-Centaurus': 31.0, 'Swat-Bahrain': 32.0,
    'Haripur': 33.0
}

class DispatchPredictor(nn.Cell):
    def __init__(self):
        super(DispatchPredictor, self).__init__()
        self.fc1 = nn.Dense(7, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(16, 2)
    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    csv_path = "field_dispatch_dataa.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Missing data file at {csv_path}")
        exit()
        
    # 📊 Load raw dataset rows
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # ⚙️ Preprocess data blocks to match 7 features
    X = np.zeros((len(df), 7), dtype=np.float32)
    X[:, 0] = df['Terrian_Complexity'].map(TERRAIN_MAP).fillna(5.0).values
    X[:, 1] = df['Weather Security'].map(WEATHER_MAP).fillna(5.0).values
    X[:, 2] = df['route_security_risk'].map(SECURITY_MAP).fillna(5.0).values
    X[:, 3] = df['ticket_urgency'].map(URGENCY_MAP).fillna(5.0).values
    X[:, 4] = df['road_visibility'].map(VISIBILITY_MAP).fillna(5.0).values
    X[:, 5] = df['engineer_Base_office'].map(CAMP_MAP).fillna(1.0).values
    X[:, 6] = df['Location'].map(LOCATION_MAP).fillna(33.0).values
    
    y_true = df['target_decision'].astype(np.int32).values
    
    # 🧠 Load trained neural network checkpoints
    net = DispatchPredictor()
    ms.load_checkpoint("mindspore_tabular_predictor.ckpt", net=net)
    
    # ⚡ Run bulk validation predictions across all data vectors
    input_tensor = Tensor(X, ms.float32)
    raw_logits = net(input_tensor).asnumpy()
    y_pred = np.argmax(raw_logits, axis=1)
    
    # 📈 Compute optimization analytics report metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📈 HUAWEI MINDSPORE ACCURACY REPORT PERFORMANCE")
    print("="*50)
    print(f"🎯 Global Model Accuracy: {acc * 100:.2f}%")
    print("-"*50)
    print("📋 CONFUSION MATRIX DISTRIBUTION LOGS:")
    print(f"   True Negative [Suspended]: {cm[0][0]}  |  False Positive: {cm[0][1]}")
    print(f"   False Negative:            {cm[1][0]}  |  True Positive [Approved]:  {cm[1][1]}")
    print("-"*50)
    print("📊 DETAIL SUMMARY DATA CLASSIFICATION PROFILE:")
    print(classification_report(y_true, y_pred, target_names=['0 (SUSPENDED)', '1 (APPROVED)']))
    print("="*50 + "\n")