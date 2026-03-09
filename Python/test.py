import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os
from keras.utils import to_categorical


def load_and_prepare_dataset(csv_file_path, images_folder_path):
    """
    Load the CSV file and prepare training/test datasets for CNN

    Parameters:
    csv_file_path (str): Path to the CSV file containing subject_id and sign_id
    images_folder_path (str): Path to the 'right' folder containing images

    Returns:
    tuple: (X_train, X_test, y_train, y_test, label_mapping)
    """

    # Load CSV file
    print("Loading CSV file...")
    df = pd.read_csv(csv_file_path)

    # Verify data structure
    print(f"Dataset shape: {df.shape}")
    print(f"Subjects: {df['subject_id'].unique()}")
    print(f"Signs: {sorted(df['sign_id'].unique())}")

    # Define sign mapping
    sign_mapping = {
        0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
        8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q',
        16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'
    }

    # Create image paths (0-based indexing for images: line 2 -> 0.jpg, etc.)
    df['image_path'] = df.index.map(lambda x: os.path.join(images_folder_path, f"{x}.jpg"))

    # Add sign letter column
    df['sign_letter'] = df.sign_id.map(lambda x: sign_mapping[x])

    print("\nDataset structure:")
    print(df.head(1000))
    print(f"\nSamples per subject per sign: {len(df) // (5 * 24)}")

    return df, sign_mapping


def load_images(image_paths, target_size=(32, 32)):
    """
    Load and preprocess images

    Parameters:
    image_paths (list): List of image file paths
    target_size (tuple): Target size for resizing images

    Returns:
    numpy.ndarray: Array of preprocessed images
    """
    images = []

    print(f"Loading {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        if i % 5000 == 0:
            print(f"Loaded {i}/{len(image_paths)} images...")

        if os.path.exists(img_path):
            # Load infrared image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size)
                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0
                # Add channel dimension for CNN (height, width, 1)
                img = np.expand_dims(img, axis=-1)
                images.append(img)
            else:
                print(f"Warning: Could not load image {img_path}")
                # Add placeholder for missing image
                images.append(np.zeros((*target_size, 1), dtype=np.float32))
        else:
                print(f"Warning: Image not found {img_path}")
                # Add placeholder for missing image
                images.append(np.zeros((*target_size, 1), dtype=np.float32))

    return np.array(images)


def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create stratified train-test split ensuring balanced distribution
    across subjects and signs

    Parameters:
    df (pandas.DataFrame): Dataset dataframe
    test_size (float): Proportion of test data
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: (train_indices, test_indices)
    """
    # Create stratification key combining subject and sign
    df['strat_key'] = df['subject_id'].astype(str) + '_' + df['sign_id'].astype(str)

    # Perform stratified split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df['strat_key'],
        random_state=random_state
    )

    return train_idx.tolist(), test_idx.tolist()


def prepare_cnn_data(csv_file_path, images_folder_path, target_size=(32, 32),
                     test_size=0.2, random_state=42):
    """
    Complete pipeline to prepare CNN training and test data

    Parameters:
    csv_file_path (str): Path to CSV file
    images_folder_path (str): Path to images folder
    target_size (tuple): Target image size
    test_size (float): Test set proportion
    random_state (int): Random seed

    Returns:
    dict: Dictionary containing all prepared data
    """

    # Load dataset
    df, sign_mapping = load_and_prepare_dataset(csv_file_path, images_folder_path)

    # Create train-test split
    print("\nCreating train-test split...")
    train_indices, test_indices = create_train_test_split(df, test_size, random_state)

    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # Split dataframes
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Load images
    print("\nLoading training images...")
    X_train = load_images(train_df['image_path'].tolist(), target_size)

    print("Loading test images...")
    X_test = load_images(test_df['image_path'].tolist(), target_size)

    # Prepare labels
    print("\nPreparing labels...")
    y_train = train_df['sign_id'].values
    y_test = test_df['sign_id'].values

    # Convert to categorical (one-hot encoding)
    num_classes = len(sign_mapping)
    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)

    # Print final statistics
    print(f"\nFinal dataset statistics:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train_categorical.shape}")
    print(f"y_test shape: {y_test_categorical.shape}")
    print(f"Number of classes: {num_classes}")

    # Check class distribution
    print("\nClass distribution in training set:")
    train_dist = np.bincount(train_df['sign_id'])
    for i, count in enumerate(train_dist):
        print(f"Sign '{sign_mapping[i]}' (id {i}): {count} samples")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_categorical,
        'y_test': y_test_categorical,
        'y_train_raw': y_train,
        'y_test_raw': y_test,
        'sign_mapping': sign_mapping,
        'train_df': train_df,
        'test_df': test_df,
        'num_classes': num_classes
    }


# Example usage
if __name__ == "__main__":
    # Set your file paths
    csv_file_path = "labels.csv"  # Replace with your CSV file path
    images_folder_path = "left"  # Replace with your images folder path

    # Prepare the dataset
    try:
        dataset = prepare_cnn_data(
            csv_file_path=csv_file_path,
            images_folder_path=images_folder_path,
            target_size=(32, 32),  # Your image size
            test_size=0.2,
            random_state=42
        )

        # Access your data
        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        sign_mapping = dataset['sign_mapping']

        print("\n✅ Dataset preparation completed successfully!")
        print("Your training and test arrays are ready for CNN training.")

        # Optional: Save the prepared data
        print("\nSaving prepared data...")
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

        print("Data saved as numpy arrays!")

    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        print("Please check your file paths and ensure all required packages are installed.")

# Required packages installation command:
# pip install pandas numpy scikit-learn opencv-python tensorflow