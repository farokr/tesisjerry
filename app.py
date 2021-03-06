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

import pickle
import base64

import io

palet = 'coolwarm'
#column = ['KOTA_LAHIR','AGAMA', 'KECAMATAN', 'ANGKATAN', 'TAHUN_RAPORT_3', 'BIDANG_STUDI_KEAHLIAN','PROGRAM_STUDI_KEAHLIAN', 'KOMPETENSI_KEAHLIAN','ASAL_SEKOLAH']
column = ['jenis kelamin', 'kota lahir', 'agama','kecamatan', 'angkatan', 'tahun raport 3', 'bidang studi keahlian','program studi keahlian', 'kompetensi keahlian', 'asal sekolah'] 

#master data kode inisialisasi
agama = pd.read_csv('agama.csv',sep=';').set_index('agama').to_dict()['inisial']
kota_kelahiran = pd.read_csv('kota_kelahiran.csv',sep=';').set_index('kota_kelahiran').to_dict()['inisial']
kecamatan = pd.read_csv('kecamatan.csv',sep=';').set_index('kecamatan').to_dict()['inisial']
tahun_angkatan = pd.read_csv('tahun_angkatan.csv',sep=';').set_index('tahun_angkatan').to_dict()['inisial']
tahun_raport = pd.read_csv('tahun_raport.csv',sep=';').set_index('tahun_raport').to_dict()['inisial']
asal_sekolah = pd.read_csv('asal_sekolah.csv',sep=';').set_index('asal_sekolah').to_dict()['inisial']
bidang_studi_keahlian = pd.read_csv('bidang_studi_keahlian.csv',sep=';').set_index('bidang_studi_keahlian').to_dict()['inisial']
program_studi_keahlian = pd.read_csv('program_studi_keahlian.csv',sep=';').set_index('program_studi_keahlian').to_dict()['inisial']
kompetensi_keahlian = pd.read_csv('kompetensi_keahlian.csv',sep=';').set_index('kompetensi_keahlian').to_dict()['inisial']

#pemrosesan data mentah menjadi data untuk proses
def proses_data(df):
    df['agama'] = df['agama'].str.lower().replace(agama)
    df['jenis kelamin']= df['jenis kelamin'].str.lower().replace({'p':1, 'l': 2})
    df['kota lahir'] = df['kota lahir'].str.lower().replace(kota_kelahiran)
    df['kecamatan'] = df['kecamatan'].str.lower().replace(kecamatan)
    df['tahun raport 3'] = df['tahun raport 3'].replace(tahun_raport)
    df['angkatan'] = df['angkatan'].replace(tahun_angkatan)
    df['bidang studi keahlian'] = df['bidang studi keahlian'].str.lower().replace(bidang_studi_keahlian)
    df['program studi keahlian'] = df['program studi keahlian'].str.lower().replace(program_studi_keahlian)
    df['kompetensi keahlian'] = df['kompetensi keahlian'].str.lower().replace(kompetensi_keahlian)
    df['asal sekolah'] = df['asal sekolah'].str.lower().replace(asal_sekolah)
    df = df.drop(['no','nama peserta didik'],axis=1)
    return df 

#fungsi link download
def get_table_download_link(df):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='utf-8', index=False, header=True, engine='xlsxwriter')
    towrite.seek(0)  # reset pointer
    #csv = df.to_csv(index=False,sep=';')
    b64 = base64.b64encode(towrite.read()).decode()
    new_filename = "datahasil.xlsx"
    href= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'

    #href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'
    return href
#end of get_table_download_link

#fungsi penampil gambar beranda
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# bagian menu DATASET SISWA    
def eda():
    st.header('Dataset Siswa')
    df = pd.read_csv('data_siswa.csv',sep=';')
    
    st.subheader('Data Awal '+str(df.shape))
    st.write(df.sample(10));

    
    if st.checkbox("Show Columns Histogram"):
        selected_columns = st.selectbox("Select Column",column)
        if selected_columns == 'asal sekolah':
            tmp = df['asal sekolah'].value_counts()
            tmp2 = pd.DataFrame({'asal sekolah':tmp.index,'JUMLAH':tmp.values})
            fig4 = plt.figure(figsize=(5,24))
            ax = sns.barplot(y='asal sekolah',x='JUMLAH',data=tmp2)
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + 0.5
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")
            st.write(fig4)
        
        elif selected_columns == 'kota lahir':
            tmp = df['kota lahir'].value_counts()
            tmp2 = pd.DataFrame({'kota lahir':tmp.index,'JUMLAH':tmp.values})
            fig4 = plt.figure(figsize=(5,12))
            ax = sns.barplot(y='kota lahir',x='JUMLAH',data=tmp2)
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + 0.5
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")
            st.write(fig4)
            
        elif selected_columns != '':
            fig4= plt.figure()
            ax = sns.countplot(x = selected_columns, data=df)
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
            plt.xticks(rotation=45,ha='right')
            st.write(fig4)
 #snd of eda           


# bagian menu K-MEANS 
def kmeans():
    st.header('K-Means')
    df_master = pd.read_csv('data_siswa.csv',sep=';')     
    df1 = df_master.copy()
    df1 = proses_data(df1)
    pca = PCA(2) #mengubah menajdi 2 kolom
    df1 = pca.fit_transform(df1) #Transform data
   
     
    st.subheader('Pemilihan nilai K Menggunakan DBI Index')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    dbi = []
    slh = []
    
    K = range(2,11)
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
     
     
    # st.text('Metode Distortion')
    # fig = plt.figure(figsize=(4,2))
    # plt.plot(K,distortions,'bx-')
    # plt.xlabel("Nilai K")
    # plt.ylabel("Distortion")
    # st.write(fig)
    
    
    # st.text('Metode Inertia')
    # fig2 = plt.figure(figsize=(4,2))
    # plt.plot(K, inertias, 'bx-')
    # plt.xlabel("Nilai K")
    # plt.ylabel('Inertia')
    # st.write(fig2)
    
    
    # st.text('Metode Silhouette')
    # fig2 = plt.figure(figsize=(4,2))
    # plt.plot(K, slh, 'bx-')
    # plt.xlabel("Nilai K")
    # plt.ylabel('Silhouette')
    # st.write(fig2)
    
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, dbi, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('DBI')
    st.write(fig2)
    st.subheader('K optimal = 7')
    
    st.header('Simulasi K-Means Model')
    k_value  = st.slider('Nilai K', min_value=2, max_value=10, step=1, value=7)
    

    model = KMeans(n_clusters=k_value,random_state=99,verbose=False) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    model.fit(df1) #proses Clustering
    label = model.predict(df1) #proses Clustering
    center = model.cluster_centers_
    
    #dibuat menjadi dataFrame
    df_master['x1'] = df1[:,0]
    df_master['y1'] = df1[:,1]
    df_master['cluster'] = label
    
    cluster = df_master['cluster'].unique()
    cluster.sort()
    
    fig3= plt.figure()
    ax = sns.scatterplot(x='x1', y='y1',hue='cluster',data=df_master,alpha=1, s=40, palette=palet)
    ax = sns.scatterplot(x=center[:, 0], y=center[:, 1],hue=range(k_value), s=100, palette=palet, ec='black',label='centroid',legend=False)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    ax = sns.countplot(x ='cluster', data=df_master,palette=palet)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = int(p.get_height())
        ax.text(_x, _y, value, ha="center")   
    st.write(fig4)
    
    st.subheader('Profil Data per Cluster')
    arr_cluster = {}
    for x in cluster:
        clu = df_master.loc[df_master['cluster'] == x]
        idx = 'cluster '+str(x)
        arr_mode = arr_cluster[idx] = {}
        arr_mode['jenis kelamin'] = clu['jenis kelamin'].mode().iat[0]
        arr_mode['kota lahir'] = clu['kota lahir'].mode().iat[0]
        arr_mode['kecamatan'] = clu['kecamatan'].mode().iat[0]
        arr_mode['agama'] = clu['agama'].mode().iat[0]
        arr_mode['angkatan'] = clu['angkatan'].mode().iat[0]
        arr_mode['tahun raport 3'] = clu['tahun raport 3'].mode().iat[0]
        arr_mode['bidang studi keahlian'] = clu['bidang studi keahlian'].mode().iat[0]
        arr_mode['program studi keahlian'] = clu['program studi keahlian'].mode().iat[0]
        arr_mode['kompetensi keahlian'] = clu['kompetensi keahlian'].mode().iat[0]
        arr_mode['asal sekolah'] = clu['asal sekolah'].mode().iat[0]

    df_cluster = pd.DataFrame(arr_cluster)
    st.dataframe(df_cluster)    

    st.subheader('Pilih Cluster')
    choice = st.selectbox("",cluster)
    res = df_master.loc[df_master['cluster'] == choice]
    st.subheader('Cluster '+str(choice)+': '+str(res.shape[0]))
    st.write(res)
    
#end of kmeans

# bagian menu K-MODES 
def kmodes():
    st.header('K-Modes')
    
    df_master = pd.read_csv('data_siswa.csv',sep=';')     
    df = df_master.copy()
    df = proses_data(df)
    pca = PCA(2) #mengubah menajdi 2 kolom
    df = pca.fit_transform(df) #Transform data
    
    
    st.subheader('Pemilihan nilai K Menggunakan DBI')
    cost = []
    slh = []
    dbi = []
    K = range(2,11)

    for k in K:
        kmode = KModes(n_clusters=k, init = "Cao", n_init = 1)
        kmode.fit_predict(df)
        #evaluasi elbow
        cost.append(kmode.cost_)
        #evaluasi silhouette
        slh.append(metrics.silhouette_score(df,kmode.labels_))
        #evaluasi DBI
        dbi.append(metrics.davies_bouldin_score(df,kmode.labels_))
        
    fig= plt.figure()
    plt.plot(K, dbi, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('DBI Score')
    plt.title('DBI Validation Method')
    st.write(fig)
    
    tmp = pd.DataFrame()
    tmp['K'] = K
    tmp['dbi'] = dbi
    st.write(tmp)
    
    
    st.subheader('K optimal = 4')
    
    st.header('Simulasi K-Modes Model')
    k_value  = st.slider('Nilai K', min_value=2, max_value=10, step=1, value=4)
    
    model = KModes(n_clusters=k_value, init = "Cao", n_init = 1, verbose=0,random_state=101)
    label = model.fit_predict(df) #proses Clustering
    center = model.cluster_centroids_
    
    #dibuat menjadi dataFrame
    df_master['x1'] = df[:,0]
    df_master['y1'] = df[:,1]
    df_master['cluster'] = label
    
    cluster = df_master['cluster'].unique()
    cluster.sort()
    
    fig3= plt.figure()
    ax = sns.scatterplot(x='x1', y='y1',hue='cluster',data=df_master,alpha=1, s=40, palette=palet)
    ax = sns.scatterplot(x=center[:, 0], y=center[:, 1],hue=range(k_value), s=100, palette=palet, ec='black',label='centroid',legend=False)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    ax = sns.countplot(x ='cluster', data=df_master,palette=palet)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = int(p.get_height())
        ax.text(_x, _y, value, ha="center")   
    st.write(fig4)
    
    st.subheader('Profil Data per Cluster')
    arr_cluster = {}
    for x in cluster:
        clu = df_master.loc[df_master['cluster'] == x]
        idx = 'cluster '+str(x)
        arr_mode = arr_cluster[idx] = {}
        arr_mode['jenis kelamin'] = clu['jenis kelamin'].mode().iat[0]
        arr_mode['kota lahir'] = clu['kota lahir'].mode().iat[0]
        arr_mode['kecamatan'] = clu['kecamatan'].mode().iat[0]
        arr_mode['agama'] = clu['agama'].mode().iat[0]
        arr_mode['angkatan'] = clu['angkatan'].mode().iat[0]
        arr_mode['tahun raport 3'] = clu['tahun raport 3'].mode().iat[0]
        arr_mode['bidang studi keahlian'] = clu['bidang studi keahlian'].mode().iat[0]
        arr_mode['program studi keahlian'] = clu['program studi keahlian'].mode().iat[0]
        arr_mode['kompetensi keahlian'] = clu['kompetensi keahlian'].mode().iat[0]
        arr_mode['asal sekolah'] = clu['asal sekolah'].mode().iat[0]

    df_cluster = pd.DataFrame(arr_cluster)
    st.dataframe(df_cluster)    

    st.subheader('Pilih Cluster')
    choice = st.selectbox("",cluster)
    res = df_master.loc[df_master['cluster'] == choice]
    st.subheader('Cluster '+str(choice)+': '+str(res.shape[0]))
    st.write(res)    


#end of kmodes

# bagian menu APLIKSI PERHITUNGAN
def apps():
    #k_value = int(st.text_input('Nilai K:',value=3))
    data = st.file_uploader("Upload a Dataset", type=["csv"])
    if data is not None:
        
        
        df = pd.read_csv(data,sep=';')
        df1 = df.copy()
        df1 = proses_data(df1)
        st.dataframe(df)
        
        pca = PCA(2) #mengubah menajdi 2 kolom
        df1 = pca.fit_transform(df1) #Transform data
        
        model = pickle.load(open('model7.pkl', 'rb'))
        label = model.predict(df1)
        center = model.cluster_centers_
        
        #dibuat menjadi dataFrame
        df['x1'] = df1[:,0]
        df['y1'] = df1[:,1]
        df['cluster'] = label
        k_value2 = len(df['cluster'].unique())
        st.write('Proses Dimulai...')
        for index, row in df.iterrows():
            st.write(str(row['no'])+'... cluster: ',str(row['cluster']))
        
        st.write('Proses Selesai')
        fig3= plt.figure()
        # sns.scatterplot(x='x1', y='y1',hue='cluster',data=df,alpha=1, s=40, palette=palet)
        # plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
        # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        ax = sns.scatterplot(x='x1', y='y1',hue='cluster',data=df,alpha=1, s=50,palette=palet)
        ax = sns.scatterplot(x=center[:, 0], y=center[:, 1],hue=range(k_value2), s=200, ec='black',palette=palet, legend=False,label = 'Centroids', ax=ax)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        st.write(fig3)

        fig4= plt.figure()
        ax = sns.countplot(x ='cluster', data=df,palette=palet)
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center")           
        st.write(fig4)
        

        cluster = df['cluster'].unique()
        cluster.sort()
        
        st.subheader('Profil Data per Cluster')
        arr_cluster = {}
        for x in cluster:
            clu = df.loc[df['cluster'] == x]
            idx = 'cluster '+str(x)
            arr_mode = arr_cluster[idx] = {}
            arr_mode['jenis kelamin'] = clu['jenis kelamin'].mode().iat[0]
            arr_mode['kota lahir'] = clu['kota lahir'].mode().iat[0]
            arr_mode['kecamatan'] = clu['kecamatan'].mode().iat[0]
            arr_mode['agama'] = clu['agama'].mode().iat[0]
            arr_mode['angkatan'] = clu['angkatan'].mode().iat[0]
            arr_mode['tahun raport 3'] = clu['tahun raport 3'].mode().iat[0]
            arr_mode['bidang studi keahlian'] = clu['bidang studi keahlian'].mode().iat[0]
            arr_mode['program studi keahlian'] = clu['program studi keahlian'].mode().iat[0]
            arr_mode['kompetensi keahlian'] = clu['kompetensi keahlian'].mode().iat[0]
            arr_mode['asal sekolah'] = clu['asal sekolah'].mode().iat[0]
    
        df_cluster = pd.DataFrame(arr_cluster)
        st.dataframe(df_cluster)
    
        st.subheader('Pilih Cluster')
        choice = st.selectbox("",cluster)
        res = df.loc[df['cluster'] == choice]
        st.subheader('Cluster '+str(choice)+': '+str(res.shape[0]))
        st.dataframe(res)
        
        st.subheader('Download')
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        
#end of apps



# bagian menu BERANDA
def home():
    #st.title('Download')
    main_bg = "home.jpg"
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .main {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
              background-position: center;
              background-repeat: no-repeat;
              background-size: contain;
        }}
        """,
        unsafe_allow_html=True
    )
    
    
#end of home  
    
def main():
    
    #ubah tanda '#' pada  activities kalau mau memasukkan/menghilangkan menu K-modes
    
    activities = ['Beranda','Dataset Siswa','K-Means','K-Modes','Aplikasi Perhitungan']
    #activities = ['Beranda','Dataset Siswa','K-Means','Aplikasi Perhitungan']

    st.sidebar.subheader("Menu")
    choice = st.sidebar.radio('',activities)
    if choice == 'Beranda':
        home()
    elif choice == 'Dataset Siswa':
        eda()
    elif choice == 'K-Means':
        kmeans()
    elif choice == 'K-Modes':
        kmodes()
    elif choice == 'Aplikasi Perhitungan':
        apps()
        

if __name__ == '__main__':
    main()