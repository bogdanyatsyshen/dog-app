import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐕")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('dog_breed_model.h5')
    return model

CLASS_NAMES = [
    'Chihuahua', 'Japanese Spaniel', 'Maltese Dog', 'Pekinese', 'Shih-Tzu',
    'Blenheim Spaniel', 'Papillon', 'Toy Terrier', 'Rhodesian Ridgeback', 'Afghan Hound',
    'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black-and-tan Coonhound',
    'Walker Hound', 'English Foxhound', 'Redbone', 'Borzoi', 'Irish Wolfhound',
    'Italian Greyhound', 'Whippet', 'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound',
    'Saluki', 'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bullterrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
    'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier', 'Norwich Terrier',
    'Yorkshire Terrier', 'Wire-haired Fox Terrier', 'Lakeland Terrier',
    'Sealyham Terrier', 'Airedale', 'Cairn', 'Australian Terrier',
    'Dandie Dinmont', 'Boston Bull', 'Miniature Schnauzer', 'Giant Schnauzer',
    'Standard Schnauzer', 'Scotch Terrier', 'Tibetan Terrier', 'Silky Terrier',
    'Soft-coated Wheaten Terrier', 'West Highland White Terrier', 'Lhasa',
    'Flat-coated Retriever', 'Curly-coated Retriever', 'Golden Retriever',
    'Labrador Retriever', 'Chesapeake Bay Retriever', 'German Short-haired Pointer',
    'Vizsla', 'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany Spaniel',
    'Clumber', 'English Springer', 'Welsh Springer Spaniel', 'Cocker Spaniel',
    'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz', 'Schipperke', 'Groenendael',
    'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old English Sheepdog',
    'Shetland Sheepdog', 'Collie', 'Border Collie', 'Bouvier Des Flandres',
    'Rottweiler', 'German Shepherd', 'Doberman', 'Miniature Pinscher',
    'Greater Swiss Mountain Dog', 'Bernese Mountain Dog', 'Appenzeller', 'Entlebucher',
    'Boxer', 'Bull Mastiff', 'Tibetan Mastiff', 'French Bulldog', 'Great Dane',
    'Saint Bernard', 'Eskimo Dog', 'Malamute', 'Siberian Husky', 'Affenpinscher',
    'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed',
    'Pomeranian', 'Chow', 'Keeshond', 'Brabancon Griffon', 'Pembroke', 'Cardigan',
    'Toy Poodle', 'Miniature Poodle', 'Standard Poodle', 'Mexican Hairless',
    'Dingo', 'Dhole', 'African Hunting Dog'
]

st.title(" Розпізнавання породи собаки")
st.markdown("Завантажте фото собаки, і штучний інтелект визначить її породу.")

try:
    with st.spinner('Завантаження нейромережі...'):
        model = load_model()
except Exception as e:
    st.error(f"Помилка завантаження моделі. Переконайтеся, що файл 'dog_breed_model.h5' знаходиться в папці з проєктом. Деталі: {e}")

file = st.file_uploader("Завантажте зображення (jpg, png)", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Завантажене фото', use_column_width=True)
    

    if st.button("Визначити породу"):
        with st.spinner('Аналізуємо зображення...'):
            img_processed = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = np.asarray(img_processed)
            img_array = (img_array.astype(np.float32) / 255.0)
            img_reshape = img_array[np.newaxis, ...]

            prediction = model.predict(img_reshape)
            predicted_index = np.argmax(prediction)
            probability = np.max(prediction)
            predicted_class = CLASS_NAMES[predicted_index]
            st.success(f"Це схоже на: **{predicted_class}**")
            st.info(f"Впевненість моделі: {probability*100:.2f}%")
            
            st.write("---")
            st.write("**Топ-3 ймовірні породи:**")

            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            for i in top_3_indices:
                st.write(f"- {CLASS_NAMES[i]}: {prediction[0][i]*100:.2f}%")


