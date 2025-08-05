from sklearn.model_selection import train_test_split

def split_data(df, target_col="SeriousDlqin2yrs", val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into train, validation, and test sets.
    """
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # first split off temp (val + test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y
    )

    # split temp into val and test
    val_fraction = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction),
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_val, y_val):
    model_name = model.__class__.__name__
    
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    recall1 = recall_score(y_val, y_pred, pos_label=1)
    f1 = f1_score(y_val, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_val, y_pred)
    
    print(f"{model_name} - Accuracy: {acc:.4f} | Recall_1: {recall1:.4f} | F1_1: {f1:.4f} | Roc_Auc: {roc_auc:.4f}")

    return {
        'model': model_name,
        'accuracy': acc,
        'recall_1': recall1,
        'f1_class1': f1,
        'roc_auc': roc_auc}




