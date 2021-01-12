import streamlit as st
from PIL import Image

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cluster import KMeans

###Loading KMeans Clustering Model######
model1=pickle.load(open('Kmeans_cluster.pkl','rb'))







def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

side_image=Image.open('Home_depot_icon.JPG')
st.sidebar.image(side_image,use_column_width=True)

image=Image.open('home_depot.JPG')
st.image(image,use_column_width=True)
icon("search")
st.sidebar.info("Example search like :bulb, cutting tool ,heater, sink , water, power etc.")
st.sidebar.info("This Application is developed by Siddhesh D. Munagekar for recommended product search in order to maximize sales in Home Depot.")

placeholder = st.empty()
placeholder.text("What can we help you find?")

searched_object = st.text_input("")


#Loading Dataset

url = 'https://drive.google.com/file/d/1ayy2Qr-DzdHZrmKKm4PC-iQ0CcmLaQ54/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
joined_df = pd.read_csv(path,encoding= 'unicode_escape')

joined_df=joined_df.dropna()


#
def run():
    ##Loading cleaned corpus file
    corpus = []

    # open file and read the content in a list
    with open('corpus.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            corpus.append(currentPlace)





    ####Feature extraction from Product Description usinng tfidf######
    vectorizer=TfidfVectorizer(stop_words='english',analyzer='word',max_features=500)

    vectorizer.fit_transform(corpus)




    ###Creating  a function to print clusters

    def print_cluster(i):

        cluster_list=[]


        #print("Cluster %d:"% i)
        for ind in ordered_centroids[i,:10]:
            #print(' %s' % terms[ind])
            cluster_list.append(terms[ind])

        print('Cluster List',cluster_list)



        return cluster_list


    k_value=10
    print("Top 10 search per clusters :")
    ordered_centroids=model1.cluster_centers_.argsort()[:,::-1]

    terms=vectorizer.get_feature_names()

    for i in range(k_value):
        print_cluster(i)




########Selecting the the cluster from the group based on user search#######

    def selected_cluster(i):
        cluster = []
        cluster.clear()
        #print("Cluster %d:" % i)
        for ind in ordered_centroids[i, :10]:
            #print(' %s' % terms[ind])
            cluster.append(terms[ind])
        return cluster

    #@st.cache(allow_output_mutation=True)
    def show_recommendations(product):
        Y = vectorizer.transform([product])
        prediction = model1.predict(Y)
        cluster = selected_cluster(prediction[0])
        return cluster


    domain = show_recommendations(searched_object)
    choice = st.radio("Select your preference", domain)

    if domain.index(choice) == 0:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)


    if domain.index(choice) == 1:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 2:
        if choice =='lithiumion':
            new_df = joined_df[joined_df['product_title'].str.contains('lithium-ion', regex=False, case=False, na=False)]
            products=new_df['product_title'].unique()
            df = pd.DataFrame(products[:10])
            df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
            st.table(df)
        else:
            new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
            products = new_df['product_title'].unique()
            df = pd.DataFrame(products[:10])
            df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
            st.table(df)

    if domain.index(choice) == 3:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 4:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 5:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 6:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 7:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)


    if domain.index(choice) == 8:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    if domain.index(choice) == 9:
        new_df = joined_df[joined_df['product_title'].str.contains(choice, regex=False, case=False, na=False)]
        products = new_df['product_title'].unique()
        df = pd.DataFrame(products[:10])
        df.rename({0: "Featuring top 10 "+choice+" related trending Products"}, axis=1, inplace=True)
        st.table(df)

    placeholder.empty()
#

if __name__=='__main__':

###This is the list of cluster items (TFIDF features) which we receive from print_cluster items
    cluster_names=['use', 'ft', 'easy', 'steel', 'used', 'design', 'feature', 'designed', 'product', 'outdoor',
'fan', 'air', 'control', 'cooking', 'filter', 'heat', 'oven', 'help', 'ft', 'room',
'paint', 'wood', 'color', 'rug', 'vary', 'surface', '65', 'seenbspproposition', 'resident', 'stain',
'battery', 'power', 'lithiumion', 'charger', 'tool', 'compact', 'protection', 'plan', 'impact', 'depot',
'tile', 'piece', 'floor', 'wall', 'flooring', 'indoor', 'commercial', 'installation', 'residential', 'case',
'door', 'glass', 'hinge', 'security', 'steel', 'opening', 'wood', 'panel', 'easy', 'lock',
'light', 'bulb', 'led', 'lighting', 'fixture', 'energy', 'incandescent', 'lamp', 'glass', 'shade',
'blade', 'cutting', 'saw', 'cut', 'steel', 'handle', 'tool', 'speed', 'motor', 'feature',
'water', 'valve', 'pipe', 'heater', 'faucet', 'toilet', 'tank', 'hot', 'pressure', 'fitting',
'shelf', 'cabinet', 'storage', 'drawer', 'vanity', 'finish', 'sink', 'hardware','design','faucet','cutting tool']


    if not searched_object:
        st.write('Please type home improvement materials in the above text box')

    elif searched_object not in cluster_names:
        st.info("Hmm...we couldn't find "+searched_object)
        st.info('Please search items related to home improvement or refer example search in the side bar')
        #exit(0)
    else:
        run()
            #searched_object=""
