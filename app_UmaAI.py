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

def main():

    st.set_page_config(layout="wide")
    #タイトルの表示
    st.title("ウマ娘 顔認識アプリ")

    #アプリの説明の表示
    st.markdown(
    '''
        ウマ娘の顔を識別するアプリです。   
        チャンミのリザルト画面に特化した学習をしてるので  
        それ以外のシーンだと上手く認識しないと思います。  
        ※ライス・カフェ・ウオッカ等顔が隠れているキャラは現状認識できないです。
    '''
    )

    #サイドバーの表示
    image = st.sidebar.file_uploader("画像をアップロードしてください", type=['jpg','jpeg', 'png'], accept_multiple_files=False)

    #サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像を使用する")
    if use_sample:
        image = "sample.jpeg"

    #保存済みのモデルをロード
    model = tf.keras.models.load_model('saved_model/my_model.h5')
    
    #画像ファイルが読み込まれた後，顔認識を実行
    if image != None:
        
        image = Image.open(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('temp.jpg',image)
        image = cv2.imread('temp.jpg')
        height, width = image.shape[:2]
        size = (width, height)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        classifier = cv2.CascadeClassifier('./lbpcascade_animeface/lbpcascade_animeface.xml')
        faces = classifier.detectMultiScale(gray_image)
        
        with open('label.pickle', 'rb') as web:
            le = pickle.load(web)

        for x,y,w,h in faces:
            #顔の部分を四角で囲む
            cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,0,255), thickness=3)
            #分類器モデルに入れるための処理
            face_image = image[y:y+h, x:x+h]
            face_image_resize = cv2.resize(face_image, (64, 64))
            X = face_image_resize.astype('float32') / 255
            #モデルの予測ラベルを取得
            label = model.predict(X.reshape(-1, 64, 64, 3))
            label = np.argmax(label, axis=1)
            label = le.inverse_transform(label)[0]
            #元画像の顔の上(x, y)地点に予測ラベルを描画
            cv2.putText(image, label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, width/500, (0, 0, 255), math.ceil(width/1000), cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.image(image, use_column_width=True)
 


if __name__ == "__main__":
    #main関数の呼び出し
    main()