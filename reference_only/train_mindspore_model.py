import os
import sys
import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

# Force terminal to output clean console strings without encoding hitches
sys.stdout.reconfigure(encoding='utf-8')

# Global Mapping Configurations for Categorical Vector Strings
TERRAIN_MAP = {"Low Complexity": 1.0, "Medium Complexity": 2.0, "High Complexity": 3.0}
WEATHER_MAP = {"Low Severity": 1.0, "Medium Severity": 2.0, "High Severity": 3.0}
RISK_MAP = {"Low Risk": 1.0, "Medium Risk": 2.0, "High Risk": 3.0}
URGENCY_MAP = {"Not urgent": 1.0, "Medium": 2.0, "Most Urgent": 3.0}
VISIBILITY_MAP = {"Clear Visibility": 1.0, "Moderate Visibility": 2.0, "Low Visibility": 3.0}

# 🗺️ LOGISTICS DICTIONARY MAP
DISTANCE_LOOKUP = {
    'islamabad office': {
        'swat-kalam': 330, 'abbottabad-pass': 130, 'islamabad-g9': 5, 'azad-kashmir': 140, 
        'lahore-dha': 380, 'haripur': 65, 'karachi-clifton': 1410, 'punjab-multan': 540, 
        'quetta-cantt': 910, 'hyderabad-latifabad': 1250, 'sukkur-barrage': 990, 'gawadar-port': 1650,
        'waziristan': 360, 'gilgit-baltistan': 510, 'skardu-valley': 600
    },
    'lahore office': {
        'swat-kalam': 610, 'abbottabad-pass': 420, 'islamabad-g9': 380, 'azad-kashmir': 430, 
        'lahore-dha': 5, 'haripur': 395, 'karachi-clifton': 1030, 'punjab-multan': 340, 
        'quetta-cantt': 720, 'hyderabad-latifabad': 870, 'sukkur-barrage': 610, 'gawadar-port': 1320,
        'waziristan': 550, 'gilgit-baltistan': 780, 'skardu-valley': 840
    },
    'karachi office': {
        'swat-kalam': 1710, 'abbottabad-pass': 1520, 'islamabad-g9': 1410, 'azad-kashmir': 1550, 
        'lahore-dha': 1030, 'haripur': 1395, 'karachi-clifton': 2, 'punjab-multan': 880, 
        'quetta-cantt': 680, 'hyderabad-latifabad': 160, 'sukkur-barrage': 480, 'gawadar-port': 620,
        'waziristan': 1100, 'gilgit-baltistan': 1900, 'skardu-valley': 1950
    }
}

class DispatchPredictor(nn.Cell):
    def __init__(self):
        super(DispatchPredictor, self).__init__()
        self.fc1 = nn.Dense(9, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Dense(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Dense(16, 2)
        
    def construct(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

def load_and_preprocess_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # Define our structural Master Template expectations
    expected_features = [
        'Terrain_Complexity', 'Weather_Security', 'route_security_risk', 
        'ticket_urgency', 'road_visibility', 'engineer_Base_office', 'Location', 'target_decision'
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == '--use-snapshot':
        csv_path = os.path.join(current_dir, 'field_dispatch_dataa_snapshot.tmp')
    else:
        csv_path = os.path.join(root_dir, 'field_dispatch_dataa.csv')
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing master database file reference at: {csv_path}")
        
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    if len(df.columns) == 1:
        s_headers = df.columns[0]
        df = df[s_headers].str.split(',', expand=True)
        df.columns = [h.strip() for h in s_headers.split(',')]

    # 🧠 AUTOMATIC SCHEMA ALIGNMENT ENGINE (NO-CODE DEPLOYMENT LAYER)
    final_column_mapping = {}
    
    for target_col in expected_features:
        # Scenario A: Exact match matches perfectly
        if target_col in df.columns:
            final_column_mapping[target_col] = target_col
            continue
            
        # Scenario B: Semantic Keyword Matching (Fuzzy mapping via string patterns)
        matched_col = None
        for raw_col in df.columns:
            if (target_col.lower() in raw_col.lower()) or (raw_col.lower() in target_col.lower()):
                matched_col = raw_col
                break
        
        # Scenario C: Statistical Distribution Signature Matching (For completely unaligned layouts)
        if not matched_col:
            for raw_col in df.columns:
                unique_vals = set(df[raw_col].dropna().astype(str).str.strip())
                
                # Identify decision targets by looking for strict binary flags (0 and 1)
                if target_col == 'target_decision' and unique_vals.issubset({'0', '1', '0.0', '1.0'}):
                    matched_col = raw_col
                    break
                # Identify branch office locations by checking for categorical patterns
                elif target_col == 'engineer_Base_office' and any('office' in val.lower() for val in unique_vals):
                    matched_col = raw_col
                    break
                    
        # Self-Healing Proxy Assignment
        if matched_col:
            print(f"🔄 Self-Healing Alignment: Automatically mapped raw column '{matched_col}' to structured framework asset '{target_col}'")
            final_column_mapping[target_col] = matched_col
        else:
            print(f"⚠️ Warning: Unstructured anomaly detected. Creating blank proxy array column for '{target_col}'")
            df[f'fallback_{target_col}'] = np.nan
            final_column_mapping[target_col] = f'fallback_{target_col}'

    # 📊 Matrix Generation Layer
    X = np.zeros((len(df), 9), dtype=np.float32)
    
    X[:, 0] = df[final_column_mapping['Terrain_Complexity']].astype(str).str.strip().map(TERRAIN_MAP).fillna(2.0).values / 3.0
    X[:, 1] = df[final_column_mapping['Weather_Security']].astype(str).str.strip().map(WEATHER_MAP).fillna(1.0).values / 3.0
    X[:, 2] = df[final_column_mapping['route_security_risk']].astype(str).str.strip().map(RISK_MAP).fillna(1.0).values / 3.0
    X[:, 3] = df[final_column_mapping['ticket_urgency']].astype(str).str.strip().map(URGENCY_MAP).fillna(2.0).values / 3.0
    X[:, 4] = df[final_column_mapping['road_visibility']].astype(str).str.strip().map(VISIBILITY_MAP).fillna(1.0).values / 3.0
    
    offices = df[final_column_mapping['engineer_Base_office']].astype(str).str.strip().str.lower().fillna('islamabad office').values
    X[:, 5] = np.where(offices == 'islamabad office', 1.0, 0.0)
    X[:, 6] = np.where(offices == 'lahore office', 1.0, 0.0)
    X[:, 7] = np.where(offices == 'karachi office', 1.0, 0.0)
    
    computed_distances = []
    for _, row in df.iterrows():
        office = str(row[final_column_mapping['engineer_Base_office']]).lower().strip()
        loc = str(row[final_column_mapping['Location']]).lower().strip()
        
        base_map = DISTANCE_LOOKUP.get(office, DISTANCE_LOOKUP['islamabad office'])
        dist = base_map.get(loc, base_map.get('gawadar-port', 250))
        computed_distances.append(dist)
        
    X[:, 8] = np.array(computed_distances, dtype=np.float32) / 1800.0
    y = df[final_column_mapping['target_decision']].fillna(0).astype(np.int32).values
    
    return X, y, X[:, 8]

def train_pangu_model():
    print("🚀 Loading matrices into deep neural optimization layers...")
    try:
        X_np, y_np, dist_vector = load_and_preprocess_data()
        
        net = DispatchPredictor()
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = nn.Adam(net.trainable_params(), learning_rate=0.005)
        
        dist_np = np.array(dist_vector, dtype=np.float32)
        
        def forward_fn(data, label, distances):
            logits = net(data)
            base_loss = loss_fn(logits, label)
            
            label_np = label.asnumpy()
            penalty_val = float(np.mean(dist_np * (label_np == 1))) * 0.45
            penalty_tensor = Tensor(penalty_val, ms.float32)
            
            return base_loss + penalty_tensor, logits
            
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        input_tensor = Tensor(X_np, ms.float32)
        label_tensor = Tensor(y_np, ms.int32)
        dist_tensor = Tensor(dist_vector, ms.float32)
        
        net.set_train()
        epochs = 60  
        
        for epoch in range(1, epochs + 1):
            (loss, logits), grads = grad_fn(input_tensor, label_tensor, dist_tensor)
            optimizer(grads)
            if epoch % 15 == 0 or epoch == epochs:
                preds = np.argmax(logits.asnumpy(), axis=1)
                accuracy = np.mean(preds == y_np) * 100
                print(f"   ↳ Epoch {epoch}/{epochs} -> Real Loss: {loss.asnumpy():.4f} -> Logical Accuracy: {accuracy:.2f}%")
                
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindspore_tabular_predictor.ckpt')
        ms.save_checkpoint(net, ckpt_path)
        print("💾 Deep network parameter checkpoint saved successfully!")
        
    except Exception as e:
        print(f"❌ Core training engine failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_pangu_model()