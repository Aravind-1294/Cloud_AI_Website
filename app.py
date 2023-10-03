import streamlit as st
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st
import requests
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings

page = st.sidebar.selectbox("Select a page", ["AI Doctor", "Lung Disease Prediction","Calorie Calculator","Food Calorie Prediction","Disease Prediction"])
if page == "AI Doctor":
    st.title("AI Doctor")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if input := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(input)
            st.session_state.messages.append({"role": "user", "content": input})
            template = """You are an AI Doctor who answers queries related to Healthcare and related things. Answer only to the questions related to Healthcare and related things, if not just reply 'Ask a Health care related question.....'
                    Human : Hi
                    Doctor : Hi, I am Your AI Doctor How can i help you?

                    Human: {query}
                    Doctor:"""
            prompt = PromptTemplate(input_variables=["query"], template=template)
            chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key='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'), prompt=prompt, verbose=True)
            output = chain.predict(query=input)
            response = f"Echo: {output}"
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
elif page =='Calorie Calculator':
    st.title("Calorie Calculator")
    st.text("Find how many calories you need to take.")
    st.sidebar.header('Enter Your Details Below for calculating your Calories intake.....')
    sex = st.sidebar.selectbox("Select Your Gender",("Male","Female"),index=None,placeholder="Select your Gender....")
    weight = st.sidebar.number_input('Weight:')
    height = st.sidebar.number_input('Height:')
    age = st.sidebar.number_input('Age (years):')
    action = st.sidebar.selectbox("Select Your activity",("sedentary","light","moderate","active"),index=None,placeholder="Select your Activity level....")

    def calculate_bmr(gender, weight, height, age):
        if gender == 'Female':
            women = (weight * 10) + (height * 6.25) - (age * 5) - 161
            return int(women)
        else:
            men = (weight * 10) + (height * 6.25) - (age * 5) + 5
            return int(men)
    value = calculate_bmr(sex,weight,height,age)

    if action =='sedentary':
        answer = value*1.25
    elif action == 'light':
        answer = value*1.375
    elif action =='moderate':
        answer = value*1.550
    elif action=='active':
        answer = value*1.725
    if st.button('Calculate'):
        st.title(f"You need to eat {answer} calories a day to maintain your current weight")
    st.text("Check out the Table below :")
    data = {
    "Gender": ["Male", "Male", "Male", "Female", "Female", "Female"],
    "Age (years)": ["19-30", "31-60", "61+", "19-30", "31-60", "61+"],
    "Calorie intake (kcal/day)": [
        "2,400-3,000", "2,200-2,800", "2,000-2,600",
        "1,800-2,400", "1,600-2,200", "1,600-2,000" ]}
    df = pd.DataFrame(data)
    st.table(df)
    st.write('Important Points to Check about calories')
    st.write("1. According to the Centers for Disease Control and Prevention (CDC), the average American adult consumes about 2,700 calories per day.")
    st.write("2. Men tend to consume more calories than women.")
    st.write("3. Calorie intake tends to decrease with age.")
    st.write("4. People who are physically active need to consume more calories than people who are sedentary.")
    st.write("5. Eating too many calories can lead to weight gain and obesity.")

elif page =='Lung Disease Prediction':
    st.title("Lung Disease Detection")
    model = tf.keras.models.load_model('Lung_disease_prediction.h5')
    image = st.file_uploader("Upload your Lung X-ray",type=['jpg','jpeg','png'])
    if st.button('predict'):
        image1 = Image.open(image)
        new_size = (224,224)
        resized_image = image1.resize(new_size)
        resized_image = np.array(resized_image)
        x = np.expand_dims(resized_image,axis=0)
        x = preprocess_input(x)
        a = np.argmax(model.predict(x))
        if a==0:
            st.write('The X-ray may have Covid-19 !!!!!')
        elif a==1:
            st.write('The X-ray is Normal and have no Lung disease.')
        else:
            st.write("The X-ray may have viral Pneumonia !!!!!")

elif page=='Disease Prediction':
    st.title("Disease Prediction")
    st.write("Know the disease from the symptoms you have")
    l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
      'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
      'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
      'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
      'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
      'enlarged_thyroid', 'brittle_nails','excessive_hunger', 'extra_marital_contacts',
      'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
      'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
      'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
      'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
      'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
      'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease',
           'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', ' Migraine',
           'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue',
           'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
           'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack',
           'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis',
           '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']

    l2 = [0] * len(l1)

    df = pd.read_csv("Training.csv")
    df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                            'Drug Reaction': 4, 'Peptic ulcer disease': 5, 'AIDS': 6, 'Diabetes ': 7,
                            'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11,
                            'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14,
                            'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                            'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                            'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27,
                            'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30,
                            'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34,
                            'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                            'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}}, inplace=True)
    X = df[l1]
    df['prognosis'] = df['prognosis'].astype(int)
    y = df[["prognosis"]]
    np.ravel(y)

    tr = pd.read_csv("Testing.csv")
    tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                            'Drug Reaction': 4, 'Peptic ulcer disease': 5, 'AIDS': 6, 'Diabetes ': 7,
                            'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11,
                            'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14,
                            'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                            'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                            'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27,
                            'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30,
                            'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34,
                            'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                            'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}}, inplace=True)
    tr['prognosis'] = tr['prognosis'].astype(int)
    X_test = tr[l1]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)

    def randomforest():
        clf4 = RandomForestClassifier()
        clf4 = clf4.fit(X, np.ravel(y))

    def predict_disease(symptoms):
        clf4 = RandomForestClassifier()
        clf4 = clf4.fit(X, np.ravel(y))

        l2 = [0] * len(l1)

        for k in range(len(l1)):
            for z in symptoms:
                if z == l1[k]:
                    l2[k] = 1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted = predict[0]

        if 0 <= predicted < len(disease):
            return disease[predicted]
        else:
            return "Not Found"

    OPTIONS = sorted(l1)

    symptom1 = st.selectbox("Symptom 1", OPTIONS)
    symptom2 = st.selectbox("Symptom 2", OPTIONS)
    symptom3 = st.selectbox("Symptom 3", OPTIONS)
    symptom4 = st.selectbox("Symptom 4", OPTIONS)
    symptom5 = st.selectbox("Symptom 5", OPTIONS)

    symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

    if st.button("Predict"):
        accuracy = randomforest()
        predicted_disease = predict_disease(symptoms)
        st.write(f"Analyzing all the symptoms, the disease might be {predicted_disease}")


elif page =="Food Calorie Prediction":
    model = tf.keras.models.load_model('calorie_three_vector_serving_three_classes.h5')

    def preprocess_image(image):
        img = Image.open(image).convert('RGB').resize((256, 256))
        img_array = np.asarray(img)
        img_array = img_array.reshape((256, 256, 3))
        return img_array

    def predict_calorie(image):
        img_array = preprocess_image(image)
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        class_probabilities = prediction[0]
        predicted_class = np.argmax(class_probabilities, axis=1)
        return predicted_class, class_probabilities
    def predict_nutri(image):
        img_array = preprocess_image(image)
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        class_probabilities = prediction[2]
        predicted_class = np.argmax(class_probabilities, axis=1)
        return predicted_class, class_probabilities
    
    st.title("Calorie Prediction App")
    st.write("Upload an image to predict calorie")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg",'jpeg'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        if st.button("Predict"):
            predicted_class, class_probabilities = predict_calorie(uploaded_file)
            pred,clas = predict_nutri(uploaded_file)

            # Display prediction results
            st.write(f"Total number of calories: {round(class_probabilities[0][0])} Cal ")
            
            st.write("Individual nutrient quantities: ")
            st.write(f"Fats: {round(clas[0][0])} g")
            st.write(f"Carbohydrates: {round(clas[0][1])} g")
            st.write(f"Proteins: {round(clas[0][2])} g")