import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

from kmodes.kmodes import KModes

column = ['KOTA_LAHIR','AGAMA', 'KECAMATAN', 'ANGKATAN', 'TAHUN_RAPORT_3', 'BIDANG_STUDI_KEAHLIAN','PROGRAM_STUDI_KEAHLIAN', 'KOMPETENSI_KEAHLIAN','ASAL_SEKOLAH']
    
    
def eda():
    st.header('Datasets')
    df = pd.read_csv('data_siswa_awal.csv',sep=';')
    
    st.subheader('Data Awal '+str(df.shape))
    st.write(df.sample(10));

    
    if st.checkbox("Show Columns Histogram"):
        selected_columns = st.selectbox("Select Column",column)
        if selected_columns == 'ASAL_SEKOLAH':
            tmp = df['ASAL_SEKOLAH'].value_counts()
            tmp2 = pd.DataFrame({'ASAL_SEKOLAH':tmp.index,'JUMLAH':tmp.values})
            fig4 = plt.figure(figsize=(5,24))
            sns.barplot(y='ASAL_SEKOLAH',x='JUMLAH',data=tmp2)
            st.write(fig4)
        
        elif selected_columns == 'KOTA_LAHIR':
            tmp = df['KOTA_LAHIR'].value_counts()
            tmp2 = pd.DataFrame({'KOTA_LAHIR':tmp.index,'JUMLAH':tmp.values})
            fig4 = plt.figure(figsize=(5,12))
            sns.barplot(y='KOTA_LAHIR',x='JUMLAH',data=tmp2)
            st.write(fig4)
            
        elif selected_columns != '':
            fig4= plt.figure()
            sns.countplot(x = selected_columns, data=df)
            plt.xticks(rotation=45,ha='right')
            st.write(fig4)
            

def kmeans():
    st.header('K-Means')
    df_master = pd.read_csv('data_siswa_awal.csv',sep=';')     
    df1 = pd.read_csv('data_ready.csv',sep=',')
   
     
    st.subheader('Pemilihan nilai K Menggunakan Elbow Method')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    dbi = []
    slh = []
    
    K = range(2,9)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(df1)
        kmeanModel.fit(df1)
        distortions.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0]
        mapping2[k] = kmeanModel.inertia_
        #evaluasi silhouette
        slh.append(metrics.silhouette_score(df1,kmeanModel.labels_))
        #evaluasi DBI
        dbi.append(metrics.davies_bouldin_score(df1,kmeanModel.labels_))
     
     
    st.text('Metode Distortion')
    fig = plt.figure(figsize=(4,2))
    plt.plot(K,distortions,'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel("Distortion")
    st.write(fig)
    
    
    st.text('Metode Inertia')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('Inertia')
    st.write(fig2)
    
    st.text('Metode DBI')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, dbi, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('DBI')
    st.write(fig2)
    
    st.text('Metode Silhouette')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, slh, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('Silhouette')
    st.write(fig2)

    st.header('K-Means Modelling')
    k_value  = st.slider('Nilai K', min_value=3, max_value=10, step=1, value=4)
    

    model = KMeans(n_clusters=k_value,random_state=99) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    label = model.fit_predict(df1) #proses Clustering
    pca = PCA(2) #mengubah menajdi 2 kolom
    dfnp = pca.fit_transform(df1) #Transform data
    center = pca.fit_transform(model.cluster_centers_)
    
    #dibuat menjadi dataFrame
    df_master['x1'] = dfnp[:,0]
    df_master['y1'] = dfnp[:,1]
    df_master['label'] = label
    
    
    fig3= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='label',data=df_master,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    sns.countplot(x ='label', data=df_master)
    st.write(fig4)
    
    cluster = df_master['label'].unique()
    cluster.sort()

    
    choice = st.selectbox("Pilih Kluster",cluster)
    res = df_master.loc[df_master['label'] == choice]
    st.subheader('Kluster '+str(choice)+' '+str(res.shape))
    st.write(res.sample(10))
     
     
     
     
def kmodes():
    st.header('K-Modes')
    df_master = pd.read_csv('data_siswa_awal.csv',sep=';')
    df = X = pd.read_csv('data_ready.csv',sep=',')
    
    cost = []
    dbi = []
    slh = []
    K = range(2,9)
 
    for k in K:
        kmode = KModes(n_clusters=k, init = "Cao", n_init = 1)
        kmode.fit_predict(df)
        cost.append(kmode.cost_)
        #evaluasi silhouette
        slh.append(metrics.silhouette_score(df,kmode.labels_))
        #evaluasi DBI
        dbi.append(metrics.davies_bouldin_score(df,kmode.labels_))
        
    
    st.text('The Elbow Method using K-modes Cost')
    fig2a = plt.figure(figsize=(4,2))
    plt.plot(K, cost, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Cost')
    st.write(fig2a)
    
    st.text('Metode DBI')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, dbi, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('DBI')
    st.write(fig2)
    
    st.text('Metode Silhouette')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, slh, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('Silhouette')
    st.write(fig2)

    st.header('K-Modes Modelling')
    
    k_value  = st.slider('Nilai K', min_value=3, max_value=10, step=1, value=4)
    model = KModes(n_clusters=k_value, init = "Cao", n_init = 1, verbose=1,random_state=99)# isnisialisasi Kmodes dgn  nilai K yg dipilih
    label = model.fit_predict(X) #proses Clustering
    
    
    pca = PCA(2) #mengubah menajdi 2 kolom
    dfnp = pca.fit_transform(X) #Transform data
    center = pca.fit_transform(model.cluster_centroids_)
    
    #dibuat menjadi dataFrame
    df_master['x1'] = dfnp[:,0]
    df_master['y1'] = dfnp[:,1]
    df_master['label'] = label
    
    
    fig3a= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='label',data=df_master,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3a)
    
    fig4a= plt.figure()
    sns.countplot(x ='label', data=df_master)
    st.write(fig4a)

    cluster2 = df_master['label'].unique()
    cluster2.sort()
    
    choice2 = st.selectbox("Pilih Kluster",cluster2)
    res = df_master.loc[df_master['label'] == choice2]
    st.subheader('Kluster '+str(choice2)+' '+str(res.shape))
    st.write(res.sample(10))
    
    
    
def main():

    activities = ['EDA','K-Means','K-Modes']	
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'EDA':
        eda()
    elif choice == 'K-Means':
        kmeans()
    elif choice == 'K-Modes':
        kmodes()
        

if __name__ == '__main__':
	main()