#必要なライブラリをインポート
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import math
import pickle
import pandas as pd
import base64
import sys
sys.path.append('./anime-face-detector-main')
import detect_anime_face as daf

#顔判別
@st.experimental_memo
def detect(image, _model, _le, namelist):
    image = Image.open(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('temp.jpg',image)
    image = cv2.imread('temp.jpg')
    height, width = image.shape[:2]
    size = (width, height)
        
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #classifier = cv2.CascadeClassifier('./lbpcascade_animeface/lbpcascade_animeface.xml')
    #faces_ori = classifier.detectMultiScale(gray_image)

    faces = daf.main(image)


    label_list = []
    label2_list = []
      
    for x,y,w,h in faces:

        #分類器モデルに入れるための処理
        face_image = image[y:y+h, x:x+h]
        face_image_resize = cv2.resize(face_image, (64, 64))
        X = face_image_resize.astype('float32') / 255
        #モデルの予測ラベルを取得
        label = _model.predict(X.reshape(-1, 64, 64, 3))
        label2 = np.where(label[0]==np.sort(label[0])[-2])
        label = np.argmax(label, axis=1)
        label = _le.inverse_transform(label)[0]
        label2 = _le.inverse_transform(label2)[0]
        label_list.append(label)
        label2_list.append(label2)
        #キャラクターカウンタを増加させる
        target_id = namelist.index(label)
        st.session_state['count{}'.format(target_id)] += 1

        #顔の部分を四角で囲む(ファルコは丸)
        if label == 'falcon' or label == 'falcon_yel':
            cen_x = math.ceil(x+w/2)
            cen_y = math.ceil(y+h/3.3)
            rad = math.ceil(w/2)
            cv2.circle(image, (cen_x,cen_y), rad, color=(0,0,255), thickness=3)
        else:
            cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,0,255), thickness=3)

        #元画像の顔の上(x, y)地点に予測ラベルを描画
        thick = max(1,math.ceil(width/800))
        cv2.putText(image, label, (x, y-30), cv2.FONT_HERSHEY_DUPLEX, width/700, (255, 255, 255), thick+1, cv2.LINE_AA)
        cv2.putText(image, label, (x, y-30), cv2.FONT_HERSHEY_DUPLEX, width/700, (0, 0, 255), thick, cv2.LINE_AA)
    
    #for x,y,w,h in faces_ori:
    #    cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,255,0), thickness=3)

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    res_list = [image, label_list, label2_list]
    
    return res_list

#第2候補への修正提案
def maybe(det_res, namelist):
    st.write("もしかして…")
    for i, name in enumerate(det_res[1]):
        name1 = det_res[1][i]
        name2 = det_res[2][i]
        col_1st, col_2nd = st.columns(2)
        with col_1st:
            st.write("{}じゃなくて{}".format(name1,name2))
        with col_2nd:
            if st.button("カウンタ修正", key=i+1000):
                i1 = namelist.index(name1)
                i2 = namelist.index(name2)
                statename1 = 'count{}'.format(i1)
                statename2 = 'count{}'.format(i2)
                st.session_state[statename1] -= 1
                st.session_state[statename2] += 1



#キャラクターカウンタの表示
def chara_counter(namelist):
    for i, name in enumerate(namelist):
        statename = 'count{}'.format(i)

        if statename not in st.session_state:
            st.session_state[statename] = 0

        col_1st, col_2nd, col_3rd , col_4th= st.columns(4)
        with col_1st:
            st.write(name)

        with col_3rd:
            if st.button("+", key=i*2):
                st.session_state[statename] += 1

        with col_4th:
            if st.button("-", key=i*2+1):
                st.session_state[statename] -= 1

        with col_2nd:
            st.write(st.session_state[statename])

#カウンタの初期化
def chara_counter_init(namelist):
    for i, name in enumerate(namelist):
        statename = 'count{}'.format(i)
        st.session_state[statename] = 0

#カウンタのcsv出力
def csv_output(namelist):
    count_list = []
    for i, name in enumerate(namelist):
        statename = 'count{}'.format(i)
        count_list.append(st.session_state[statename])
    df = pd.DataFrame((zip(namelist, count_list)), columns = ['Name', 'Count'])
    csv = df.to_csv(index=False) 

    return(csv)




def main():

    st.set_page_config(layout="wide")

    #最初にキャッシュを消去
    if 'init' not in st.session_state: 
        st.session_state['init'] = 0 
        detect.clear()
    

    #タイトルの表示
    st.title("ウマ娘 顔認識アプリ")

    #アプリの説明の表示
    st.markdown(
    '''
        ウマ娘の顔を識別するアプリです。   
        チャンミのリザルト画面に特化した学習をしてるので  
        それ以外のシーンだと上手く認識しないと思います。  
    '''
    )

    #サイドバーの表示：単一画像
    st.sidebar.markdown('''### 単一画像アップロード''')
    image = st.sidebar.file_uploader('こちらから画像をアップロードしてください',type=['jpg','jpeg', 'png'], accept_multiple_files=False)

    #サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像を使用する")
    if use_sample:
        image = "sample.jpeg"
    st.sidebar.markdown('')
    st.sidebar.markdown('')

    #サイドバーの表示：複数画像
    st.sidebar.markdown('''### 複数画像アップロード''')
    image_m = st.sidebar.file_uploader("複数画像の場合はこちら", type=['jpg','jpeg', 'png'], accept_multiple_files=True)

    #保存済みのモデルをロード
    model = tf.keras.models.load_model('saved_model/my_model.h5')
    with open('label.pickle', 'rb') as web:
        le = pickle.load(web)

    #ネームリストのロード
    namelist = pd.read_csv('umanames.csv',index_col=False, header=None)
    namelist = list(namelist[0])

    #複数の画像ファイルが読み込まれた後，そのうちの一つを認識対象とする
    if len(image_m) != 0 :
        if 'page' not in st.session_state:
            st.session_state['page'] = 0
        st.markdown('')
        col_1st, col_2nd, col_3rd = st.columns(3)
        with col_1st:
            if st.button("<"):
                if st.session_state['page'] != 0:
                    st.session_state['page'] -= 1

        with col_3rd:
            if st.button(">"):
                if st.session_state['page'] != len(image_m)-1:
                    st.session_state['page'] += 1

        with col_2nd:
            if st.session_state['page'] > len(image_m)-1:
                st.session_state['page'] = len(image_m)-1
            st.markdown(f''' {st.session_state['page']+1}/{len(image_m)}''')

        st.markdown(f''' 【{image_m[st.session_state['page']].name}】''')
        image = image_m[st.session_state['page']]

    #画像ファイルが読み込まれた後，顔認識を実行し結果を表示
    if image != None:
        det_res = detect(image, model, le, namelist)
        image = det_res[0]
        h, w = image.shape[:2]
        if w > 500:
            width = 500
            height = round(h * (width / w))
            image = cv2.resize(image, dsize=(width, height))
        st.image(image, use_column_width='auto')
        maybe(det_res, namelist)

    else:
        st.markdown("""#### 集計のやり方""")
        st.markdown(
        '''
            1. 集計する画像をまとめてアップロード  
            2. 表示されている認識結果が  
                2-1. 正しい場合はそのまま3へ  
                2-2. 間違っている場合、「もしかして…」で提案される修正が合っていれば[カウンタ修正]を押下  
                2-3. 「もしかして…」で提案される修正も間違っていればカウンタ一覧の[+][-]で手動で修正(すみません…)  
            3. [>]を押して次の画像へ  
            4. すべての画像が終わったらサイドバーの[Download]から結果を出力
        '''
        )



    #カウンタの見出しとクリアボタン
    st.markdown('')
    st.markdown('')
    col_left, col_right= st.columns(2)
    with col_left:
            st.markdown("""### カウンタ""")

    with col_right:
        if st.button("All Clear"):
            detect.clear()
            chara_counter_init(namelist)

    #カウンタの説明の表示
    st.markdown(
    '''
        検出したウマ娘をカウントします。
        検出結果が間違っている場合はボタンを押すことで手動で修正が行えます。  
    '''
    )

    #キャラクターカウンタの生成
    chara_counter(namelist)

    #カウント結果のダウンロード
    csv = csv_output(namelist)  
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.csv">Download</a>'
    st.sidebar.markdown('')
    st.sidebar.markdown('')
    st.sidebar.markdown('''### カウント結果のダウンロード''')
    st.sidebar.markdown(f" {href}", unsafe_allow_html=True)
 




if __name__ == "__main__":
    #main関数の呼び出し
    main()