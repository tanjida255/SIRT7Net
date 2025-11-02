import os,pandas as pd,numpy as np,networkx as nx,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

TAR_FILE="tar.csv"
PPI_FILE="ppi.csv"
OUTPUT_DIR="results"
TOP_N=10
EPOCHS=1000
LR=0.01
PLOT_DPI=300
HEATMAP_TOP_N=50

class Data:
    def __init__(self,x=None,edge_index=None,edge_attr=None,y=None):
        self.x = x
        self.edge_index = edge_index if edge_index is not None else torch.zeros((2,0),dtype=torch.long)
        self.edge_attr = edge_attr if edge_attr is not None else torch.zeros((self.edge_index.size(1),),dtype=torch.float)
        self.y = y
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

class GraphConv(nn.Module):
    def __init__(self,in_channels,out_channels,bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels,out_channels,bias=bias)
    def forward(self,x,edge_index,edge_attr=None):
        N = x.size(0)
        E = edge_index.size(1) if edge_index is not None else 0
        x_trans = self.linear(x)
        if E == 0:
            return x_trans
        device = x.device
        src = edge_index[0]
        dst = edge_index[1]
        if edge_attr is None or edge_attr.numel() == 0:
            w = torch.ones((E,),dtype=x.dtype,device=device)
        else:
            w = edge_attr.to(device).to(dtype=x.dtype)
        deg = torch.zeros(N,dtype=x.dtype,device=device)
        deg = deg.scatter_add(0,dst,w)
        eps = 1e-12
        deg_inv_sqrt = (deg+eps).pow(-0.5)
        src_scale = deg_inv_sqrt[src].unsqueeze(1)
        dst_scale = deg_inv_sqrt[dst].unsqueeze(1)
        w_unsq = w.unsqueeze(1)
        x_src = x_trans[src]
        messages = x_src * (w_unsq * src_scale * dst_scale)
        out = torch.zeros_like(x_trans)
        out = out.index_add(0,dst,messages)
        return out

GCNConv = GraphConv

def load_exp(f):
    df=pd.read_csv(f)
    if len(df.columns)==1:
        df.columns=['Gene']; df['Value']=1.0
    elif 'Target' in df.columns and 'Value' in df.columns:
        df=df[['Target','Value']]; df.columns=['Gene','Value']
    else:
        df.columns=list(df.columns[:2]); df=df[[df.columns[0],df.columns[1]]]; df.columns=['Gene','Value']
    return df.dropna().reset_index(drop=True)

def load_ppi(f):
    df=pd.read_csv(f)
    if not set(['node1','node2','combined_score']).issubset(df.columns):
        df.columns=['node1','node2']+list(df.columns[2:])
    df=df[['node1','node2','combined_score']]; df.columns=['Gene1','Gene2','Combined_Score']
    return df.dropna().reset_index(drop=True)

def make_features(exp_df,ppi_df):
    G=nx.Graph()
    for _,r in ppi_df.iterrows(): G.add_edge(r.Gene1,r.Gene2,weight=r.Combined_Score)
    genes=set(exp_df.Gene.unique())|set(G.nodes())
    for g in genes:
        if g not in G: G.add_node(g)
    deg=nx.degree_centrality(G); clo=nx.closeness_centrality(G)
    bet=nx.betweenness_centrality(G); clu=nx.clustering(G)
    try: ecc=nx.eccentricity(G)
    except: ecc={n:0 for n in G}
    avg_score={n:(sum(G[n][nbr]['weight'] for nbr in G[n])/len(list(G[n])) if list(G[n]) else 0.0) for n in G.nodes()}
    rows=[]
    for g in genes:
        val = exp_df.loc[exp_df.Gene==g,'Value'].mean() if not exp_df.loc[exp_df.Gene==g].empty else 0.0
        rows.append({'Gene':g,'Value':val,'Degree':deg.get(g,0),'Closeness':clo.get(g,0),
                     'Betweenness':bet.get(g,0),'ClusteringCoefficient':clu.get(g,0),
                     'Eccentricity':ecc.get(g,0),'AvgCombinedScore':avg_score.get(g,0)})
    return pd.DataFrame(rows), G

class Model(nn.Module):
    def __init__(s,in_dim,hd=64):
        super().__init__()
        s.conv1=GCNConv(in_dim,hd); s.conv2=GCNConv(hd,hd)
        s.fc1=nn.Linear(hd,hd); s.fc2=nn.Linear(hd,hd//2); s.fc3=nn.Linear(hd//2,1)
        s.drop=nn.Dropout(0.2)
    def forward(s,x,edge_index,edge_attr):
        x=F.relu(s.conv1(x,edge_index,edge_attr)); x=s.drop(x)
        x=F.relu(s.conv2(x,edge_index,edge_attr))
        x=F.relu(s.fc1(x)); x=s.drop(x); x=F.relu(s.fc2(x)); x=s.fc3(x)
        return torch.sigmoid(x)

def plot_and_save_train_val(losses,val_losses,train_r2,val_r2,out_dir):
    epochs = list(range(1,len(losses)+1))
    plt.figure(figsize=(8,6))
    plt.plot(epochs,losses,label='Train Loss',color='blue')
    plt.plot(epochs,val_losses,label='Val Loss',color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train vs Val Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"loss_train_val.png"),dpi=PLOT_DPI); plt.close()
    plt.figure(figsize=(8,6))
    plt.plot(epochs,train_r2,label='Train R2',color='blue')
    plt.plot(epochs,val_r2,label='Val R2',color='orange')
    plt.xlabel('Epoch'); plt.ylabel('R2'); plt.title('Train vs Val R2'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"r2_train_val.png"),dpi=PLOT_DPI); plt.close()

def save_metrics_csv(metrics_dict,out_dir):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(os.path.join(out_dir,"metrics_splits.csv"))

def train(features,G,epochs=EPOCHS,lr=LR,out_dir=OUTPUT_DIR):
    feats=['Value','Degree','Betweenness','Closeness','ClusteringCoefficient','Eccentricity','AvgCombinedScore']
    scaler=StandardScaler()
    comp = (features['Value']*0.15 + features['Degree']*0.15 + features['Betweenness']*0.15 +
            features['Closeness']*0.15 + features['ClusteringCoefficient']*0.1 + features['Eccentricity']*0.1 +
            features['AvgCombinedScore']*0.2)
    features['Importance']=1/(1+np.exp(-scaler.fit_transform(comp.values.reshape(-1,1)).ravel()))
    x=torch.tensor(features[feats].values,dtype=torch.float)
    y=torch.tensor(features['Importance'].values,dtype=torch.float).view(-1,1)
    idx={g:i for i,g in enumerate(features.Gene)}
    edges=[]; attrs=[]
    for u,v,d in G.edges(data=True):
        if u in idx and v in idx:
            edges += [[idx[u],idx[v]],[idx[v],idx[u]]]; attrs += [d['weight'],d['weight']]
    edge_index = torch.tensor(edges,dtype=torch.long).t() if edges else torch.zeros((2,0),dtype=torch.long)
    edge_attr = torch.tensor(attrs,dtype=torch.float) if attrs else torch.zeros(0,dtype=torch.float)
    data=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)
    n=data.x.size(0)
    ids=list(range(n))
    tr, tmp = train_test_split(ids, train_size=0.6, random_state=1)
    val, test = train_test_split(tmp, train_size=0.5, random_state=1)
    mask = lambda ids: torch.tensor([i in ids for i in range(n)],dtype=torch.bool)
    data.train_mask, data.val_mask, data.test_mask = mask(tr), mask(val), mask(test)
    m=Model(len(feats)); opt=torch.optim.Adam(m.parameters(),lr=lr); loss_fn=nn.MSELoss()
    train_losses=[]; val_losses=[]
    train_r2=[]; val_r2=[]
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out=m(data.x,data.edge_index,data.edge_attr)
        loss=loss_fn(out[data.train_mask],data.y[data.train_mask])
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out_all = m(data.x,data.edge_index,data.edge_attr).cpu().numpy().ravel()
            y_all = data.y.cpu().numpy().ravel()
            tr_idx = data.train_mask.numpy()
            val_idx = data.val_mask.numpy()
            y_tr = y_all[tr_idx]; p_tr = out_all[tr_idx]
            y_val = y_all[val_idx]; p_val = out_all[val_idx]
            tr_loss = mean_squared_error(y_tr,p_tr) if len(y_tr)>0 else float('nan')
            val_loss = mean_squared_error(y_val,p_val) if len(y_val)>0 else float('nan')
            tr_r = r2_score(y_tr,p_tr) if len(y_tr)>1 else float('nan')
            val_r = r2_score(y_val,p_val) if len(y_val)>1 else float('nan')
            train_losses.append(tr_loss); val_losses.append(val_loss)
            train_r2.append(tr_r); val_r2.append(val_r)
    m.eval()
    with torch.no_grad():
        preds_all = m(data.x,data.edge_index,data.edge_attr).cpu().numpy().ravel()
    features['PredictedScore']=preds_all
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    metrics = {}
    split_names = {'train':data.train_mask,'val':data.val_mask,'test':data.test_mask}
    for name,mask_t in split_names.items():
        mask_np = mask_t.numpy()
        y_true = data.y.cpu().numpy().ravel()[mask_np]
        y_pred = preds_all[mask_np]
        if len(y_true)==0:
            mse = float('nan'); rmse = float('nan'); r2 = float('nan')
        else:
            mse = mean_squared_error(y_true,y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true,y_pred) if len(y_true)>1 else float('nan')
        metrics[name] = {'MSE':mse,'RMSE':rmse,'R2':r2}
    save_metrics_csv(metrics,out_dir)
    plot_and_save_train_val(train_losses,val_losses,train_r2,val_r2,out_dir)
    return features, m, data, metrics

def plot_centrality_heatmap(features,G,out_dir,top_n=HEATMAP_TOP_N):
    nodes = list(G.nodes())
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    bet = nx.betweenness_centrality(G)
    clu = nx.clustering(G)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except:
        eig = {n:0.0 for n in nodes}
    centrality_df = pd.DataFrame({
        'Gene': nodes,
        'Degree': [deg.get(n,0.0) for n in nodes],
        'Closeness': [clo.get(n,0.0) for n in nodes],
        'Betweenness': [bet.get(n,0.0) for n in nodes],
        'ClusteringCoefficient': [clu.get(n,0.0) for n in nodes],
        'Eigenvector': [eig.get(n,0.0) for n in nodes]
    })
    merged = centrality_df.merge(features[['Gene','PredictedScore']],on='Gene',how='left').fillna(0)
    merged = merged.sort_values('PredictedScore',ascending=False).head(top_n)
    cols = ['Degree','Closeness','Betweenness','ClusteringCoefficient','Eigenvector']
    mat = merged[cols].values
    eps = 1e-12
    mat_min = mat.min(axis=0)
    mat_max = mat.max(axis=0)
    mat_norm = (mat - mat_min) / (mat_max - mat_min + eps)
    fig, ax = plt.subplots(figsize=(10, max(4,0.2*len(merged))))
    cax = ax.imshow(mat_norm, aspect='auto', cmap='Reds', interpolation='nearest')
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged['Gene'].values, fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            val = mat_norm[i,j]
            txt = f"{mat[i,j]:.2f}"
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=7)
    fig.colorbar(cax, ax=ax, fraction=0.03)
    ax.set_title('Centrality Metrics for Proteins')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"centrality_heatmap_red.png"),dpi=PLOT_DPI)
    plt.close()

def identify(exp_file,ppi_file,top=TOP_N,out=OUTPUT_DIR):
    if not os.path.exists(out): os.makedirs(out)
    exp=load_exp(exp_file); ppi=load_ppi(ppi_file)
    feats,G=make_features(exp,ppi)
    preds,model,data,metrics=train(feats,G,epochs=EPOCHS,lr=LR,out_dir=out)
    top_df=preds.sort_values('PredictedScore',ascending=False).head(top)
    top_df.to_csv(os.path.join(out,"top_targets.csv"),index=False)
    plot_centrality_heatmap(preds,G,out,top_n=HEATMAP_TOP_N)
    return top_df, model, G, metrics

if __name__=="__main__":
    top_df,model,G,metrics=identify(TAR_FILE,PPI_FILE,TOP_N,OUTPUT_DIR)
    print(metrics)
    print(top_df[['Gene','PredictedScore']].head(20))
