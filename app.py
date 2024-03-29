import pickle
import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
from rembg import remove
from io import BytesIO
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

model = pickle.load(open('fruits_classification_model2.pkl', 'rb'))
scaler = pickle.load(open('scalerinstance.pkl', 'rb'))

# Dictionary to map indices to fruit names
fruit_mapping = {
    0: 'Banana',
    1: 'Coconut',
    2: 'Peach',
    3: 'Pineapple'
}

def remove_transparency(source, background_color):
    source_img = cv2.cvtColor(source[:,:,:3], cv2.COLOR_BGR2GRAY)
    source_mask = source[:,:,3]  * (1 / 255.0)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))

def perform_inference(uploaded_image, remove_background):
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

    if remove_background:
        image = remove(image)
        trans_mask = image[:,:,3] == 0
        image = image.copy()
        image[trans_mask] = [255, 255, 255, 255]
        temp_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        new_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    else:
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(new_img, (100, 100))
    flattened_image = resized_image.flatten()
    scaled_image = scaler.transform([flattened_image])
    predicted_index = model.predict(scaled_image)[0]
    predicted_fruit = fruit_mapping.get(predicted_index, 'Unknown')

    return predicted_fruit, new_img

def eda_section():
    st.sidebar.subheader("EDA")
    st.sidebar.write("This section highlights the step that we took while performing Exploratory Data Analysis. It displays the dataset images and color distributions mainly.")

    st.subheader("EDA")

    st.write("First, we checked the number of training and testing samples of all our fruits.")

    st.code("""
    train_class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    test_class_counts = dict(zip(*np.unique(y_test, return_counts=True)))

    print("Training samples per class:")
    for label, count in train_class_counts.items():
        print(f"Class {label}: {count} samples")

    print("\nTesting samples per class:")
    for label, count in test_class_counts.items():
        print(f"Class {label}: {count} samples")
    """, language="python")

    st.write("Training samples per class:")
    st.info("Class 0: 490 samples - Class 1: 490 samples - Class 2: 492 samples - Class 3: 490 samples", icon="ℹ️")

    st.write("Testing samples per class:")
    st.info("Class 0: 166 samples - Class 1: 166 samples - Class 2: 164 samples - Class 3: 166 samples", icon="ℹ️")

    st.write("Then, we displayed the color distribution of the images per class.")

    st.code("""
    plot_color_distribution(X_train, y_train, fruits)
    """, language="python")

    st.image('colordistribution.png', caption='Color Distribution', use_column_width=True)

    st.write("Considering the similarity of the Coconut and Pineapple color distribution, we visualized their instances to see if similarities appeared to be evident to the naked eye.")

    st.code("""
    def plot_image_grid(images, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(images[i])
            ax.axis('off')
        plt.show()

    print(fruits[y_train[490]])
    plot_image_grid(X_train[490:590], 10, 10)

    print(fruits[y_train[1472]])
    plot_image_grid(X_train[1472:1962], 10, 10)
    """, language="python")

    st.image('coconutgrid.png', caption='Coconut Instances', use_column_width=True)
    st.image('pineapplegrid.png', caption='Pineapple Instances', use_column_width=True)

    st.info("As we can see from the images of Coconut and Pineapple, they appear to be very similar, which suggests that this classification task will not be easy and thus we would likely need to use a non-linear kernel to separate the classes well.", icon="ℹ️")

def processing_training_section():
    st.sidebar.subheader("Processing/Training")
    st.sidebar.write("This section highlights the steps that we undertook when pre-processing the data and subsequently training our model. Code is also provided.")

    st.subheader("Processing/Training")

    st.write("First, we retrieved all the images from our dataset and placed them into appropriate training and test sets.")

    st.code("""
    def getYourFruits(fruits, data_type):
        data = []
        labels = []
        
        for fruit in fruits:
            folder_path = os.path.join('/content/drive/MyDrive/fruits-360', data_type, fruit)
            label = fruits.index(fruit)
            
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    data.append(image)
                    labels.append(label)

        return np.array(data), np.array(labels)

    fruits = ['Banana', 'Cocos', 'Peach', 'Pineapple']

    X_train, y_train = getYourFruits(fruits, 'Training')
    X_test, y_test = getYourFruits(fruits, 'Test')
    """, language="python")

    st.write("Then, we flattened the array of pixels into a 1D array, instead of a 2D array, and applied StandardScaler to normalize the features.")

    st.code("""
    scaler = StandardScaler()

    flattened_X_train = [image.flatten() for image in X_train]
    X_train_scaled = scaler.fit_transform(flattened_X_train)

    flattened_X_test = [image.flatten() for image in X_test]
    X_test_scaled = scaler.transform(flattened_X_test)
    """, language="python")

    st.write("We then trained our model using an SVM classifier with the non-linear rbf kernel.")

    st.code("""
    model = SVC(gamma='auto', kernel='rbf')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    """, language="python")

    st.write("Finally, we evaluated the model and visualized the results.")

    st.code("""
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import confusion_matrix
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    precision = metrics.accuracy_score(y_pred, y_test) * 100
    cm = confusion_matrix(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=model.classes_)
    y_pred_bin = label_binarize(y_pred, classes=model.classes_)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(model.classes_.shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 7))
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)

    for i in range(model.classes_.shape[0]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
            
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = np.unique(y_pred)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    """, language="python")

    st.image('classification_report.png', caption='Classification Report')
    st.image('roccurve.png', caption='ROC Curve')
    st.image('confusion_matrix.png', caption='Confusion Matrix')

    st.write("After this, we packaged our model and the scaler instance to be used during inference.")

    st.code("""
    import pickle
    with open('fruits_classification_model2.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open('scalerinstance.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    """, language="python")

def perform_batch_inference(uploaded_images, remove_background):
    predictions = []

    for uploaded_image in uploaded_images:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        if remove_background:
            image = remove(image)
            trans_mask = image[:,:,3] == 0
            image = image.copy()
            image[trans_mask] = [255, 255, 255, 255]
            temp_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            new_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        else:
            new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized_image = cv2.resize(new_img, (100, 100))
        flattened_image = resized_image.flatten()
        scaled_image = scaler.transform([flattened_image])
        predicted_index = model.predict(scaled_image)[0]
        predicted_fruit = fruit_mapping.get(predicted_index, 'Unknown')
        predictions.append((predicted_fruit, new_img))

    return predictions

def generate_recipe(fruits):
    fruits_str = ",".join(str(element) for element in fruits)
    original_string = """Fruit(s): {text}

    Above provided are the fruits (or a single fruit) that I have. I want to create a cocktail out of these fruits. Please write me a recipe for that cocktail, making it detailed enough with how many proportions I need of everything etc. Give me the name of the cocktail at the start as well.
    The format of your response should be the following. Don't write anything else in your response. Make sure the headings ('Cocktail', 'Ingredients', 'Recipe') are bold and the steps are numbered and the ingredients are in bullet points.:
    Cocktail:<Insert name here>
    
    Ingredients:
    <Insert a list of ingredients here>
    
    Recipe:
    <Insert the steps to prepare that cocktail. Make sure they are detailed enough with how many proportions I need of everything etc>"""

    prompt = original_string.replace("{text}", fruits_str)

    PAT = st.secrets["pat"]
    USER_ID = 'openai'
    APP_ID = 'chat-completion'
    MODEL_ID = st.secrets["mid"]
    MODEL_VERSION_ID = st.secrets["mvid"]
    RAW_TEXT = prompt

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=RAW_TEXT
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

    output = post_model_outputs_response.outputs[0]

    result = output.data.text.raw

    return result

def inference_section():
    st.sidebar.subheader("Inference")
    st.sidebar.write("This section displays the application of our model by giving a recipe for a cocktail from the predicted fruits.")
    st.title("Cocktail Recipe Generator - ML Project")
    st.subheader("Using SVM and GPT4", divider="rainbow")
    st.write("Input images of fruits (Bananas, Coconuts, Peaches, Pineapples) and get a cocktail recipie involving those fruits.")

    uploaded_images = st.file_uploader(label="Choose multiple files", type=['png', 'jpg'], accept_multiple_files=True)
    remove_background = st.checkbox("Remove Background", value=False)

    if uploaded_images:
        if st.button('Predict'):
            predictions = perform_batch_inference(uploaded_images, remove_background)
            
            for i, (predicted_fruit, image) in enumerate(predictions):
                st.success(f'Image {i + 1}: This is a {predicted_fruit}')
                st.image(image, caption=f'Uploaded Image {i + 1}')

            st.divider()

            if predictions:
                st.subheader(":tropical_drink: Cocktail recipe:", divider="rainbow")
                recipe = generate_recipe(predictions)
                st.write(recipe)


def main():
    selected_section = st.sidebar.radio("Select a section", ["EDA", "Processing/Training", "Inference"])

    if selected_section == "EDA":
        eda_section()
    elif selected_section == "Processing/Training":
        processing_training_section()
    elif selected_section == "Inference":
        inference_section()

if __name__ == '__main__':
    main()
