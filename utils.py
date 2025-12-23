import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from graphviz import Digraph, Source
from sklearn.metrics import accuracy_score, f1_score

def show_missing_values(df):
    def min_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(round(df[col].min(), 2))
        return np.nan
    def max_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(round(df[col].max(), 2))
        return np.nan

    missing_df = pd.DataFrame({
        'S. No.': range(1, len(df.columns) + 1),
        'Column Name': df.columns,
        'Min': [min_or_nan(col) for col in df.columns],
        'Max': [max_or_nan(col) for col in df.columns],
        'n Unique': df.nunique(),
        'NaN count': df.isna().sum(),
        'NaN percentage': (df.isna().mean() * 100).round(3).astype(str) + '%',
        'dtype': df.dtypes.astype(str),

    }).set_index('S. No.')

    unique_dtypes = missing_df['dtype'].unique()
    palette = sns.color_palette("Set2", n_colors=len(unique_dtypes))
    dtype_color_map = {dt: f"background-color: {mcolors.to_hex(color)}" for dt, color in zip(unique_dtypes, palette)}

    def color_row(row):
        return [dtype_color_map.get(row['dtype'], "")] * len(row)

    return missing_df.style.apply(color_row, axis=1)

def show_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers


def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, df, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def group_rare_categories(series, threshold=0.03, other_label="Other"):
    if pd.api.types.is_categorical_dtype(series):
        series = series.astype("object")

    category_counts = series.value_counts(normalize=True, dropna=False)
    cumulative_distribution = category_counts.cumsum()

    rare_categories = category_counts[cumulative_distribution > (1 - threshold)].index
    return series.replace(list(rare_categories), other_label)

def binning(series, bins=4):
    nunique = series.nunique()
    if nunique <= bins:
        return series.astype('category')
    return pd.qcut(series, bins, duplicates='drop').astype('category')

def cramers_v_matrix(df):
    df_ = df.dropna().drop_duplicates()
    df_cat = df_.select_dtypes(include=['object', 'category', 'bool']).copy()
    df_num = df_.select_dtypes(include=['number']).copy()
    matrix = pd.DataFrame(index=df_.columns, columns=df_.columns, dtype=float)
    bool_cols = df_.select_dtypes(include=['bool']).columns
    df_[bool_cols] = df_[bool_cols].astype('category')
    not_bool_cols = df_.columns.difference(bool_cols)

    for col in not_bool_cols:
        df_[col] = group_rare_categories(df_[col], threshold=0.03)

    if not df_num.empty:
        df_num = df_num.apply(binning, bins=4)
    
    df_combined = pd.concat([df_cat, df_num], axis=1)
    matrix = pd.DataFrame(index=df_combined.columns, columns=df_combined.columns, dtype=float)

    for i, col1 in enumerate(df_combined.columns):
        for j, col2 in enumerate(df_combined.columns):
            if i == j:
                matrix.loc[col1, col2] = 1.0
            elif i > j:
                 matrix.loc[col1, col2] = cramers_v(df_combined[col1], df_combined[col2])
            else:
                matrix.loc[col1, col2] = np.nan
    return matrix

def correlation_heatmap(matrix, title=None, cmap='Oranges', min=None):
    num_vars = len(matrix)    
    fig_width = max(8, num_vars * 0.6)
    fig_height = max(6, num_vars * 0.6)   
    plt.figure(figsize=(fig_width, fig_height))  
    ax = sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        linewidths=1, 
        linecolor="white", 
        annot_kws={"size": max(8, 15 - num_vars * 0.5)},
        cbar_kws={"shrink": 0.5, "aspect": 20},
        square=True,
        mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
        vmin=min
    )  
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_facecolor('white')
    return ax

def cat_corr_heatmap(matrix):
    ax = correlation_heatmap(matrix, title="Cramers V", cmap='Blues', min=0)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"])
    plt.show()

def num_corr_heatmap(matrix):
    ax = correlation_heatmap(matrix, title="Pearson", cmap='coolwarm', min=-1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "1"])
    plt.show()

def plot_confusion_matrix(
    cm,
    classes,
    title="Confusion matrix",
    ax=None,
    cmap="Oranges",
    fmt="d",
    xlabel="Predicted",
    ylabel="True"
):
    """
    Plottet eine Konfusionsmatrix.
    """

    show = False
    if ax is None:
        _, ax = plt.subplots()
        show = True

    sns.heatmap(
        cm,
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        cmap=cmap,
        fmt=fmt,
        cbar=False,
        ax=ax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()



def skfcv(model, X, y, k=5, metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], train_score=False, random_seed=42, cm=True):
    """
    Berechnet Metriken mittels stratified k-Fold Kreuzvalidierung und Konfusionsmatrix mit cross_val_predict.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    cv = cross_validate(model, X, y, cv=skf, scoring=metrics, return_train_score=train_score)
    if not cm:
        return dict(cv=cv)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    labels = np.unique(y_pred)
    cm = confusion_matrix(y, y_pred, labels=labels)
    return dict(cv=cv, cm=cm)

def pipeline(df, features):
    """
    Erstellt einen Pipeline-Datensatz, indem fehlende Werte in den angegebenen Merkmalen entfernt werden.
    """
    df_copy = df.copy(deep=True)
    df_copy = df_copy.dropna(subset=features)
    return df_copy[features], df_copy['species']

def loocv(model, X, y, metrics=['accuracy'], train_score=False):
    """
    Führt Leave-One-Out Kreuzvalidierung durch und berechnet die angegebenen Metriken.
    """
    loo = LeaveOneOut()
    return cross_validate(model, X, y, cv=loo, scoring=metrics, return_train_score=train_score)

def print_cv(cv, time=False):
    """
    Gibt die Mittelwerte und Standardabweichungen der Kreuzvalidierungsergebnisse aus.
    """
    for key, values in cv.items():
        if not time and 'time' in key:
            continue
        print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")

def plot_dt(dtc):
    """
    Visualisiert einen Entscheidungsbaum.
    """
    dot_data = export_graphviz(
        dtc,
        out_file=None,
        feature_names=dtc.feature_names_in_,
        class_names=dtc.classes_,
        filled=True,
        rounded=True,
        special_characters=True
    )
    return Source(dot_data)

def get_leaf_masks(X): # generiert von ChatGPT (Stand: 2025-12)
    bl = X["bill_length_mm"]
    bd = X["bill_depth_mm"]

    mask0 = (bl <= 43.25) & (bd <= 14.8)
    mask1 = (bl <= 43.25) & (bd >  14.8)
    mask2 = (bl >  43.25) & (bd <= 16.45)
    mask3 = (bl >  43.25) & (bd >  16.45)

    return {0: mask0, 1: mask1, 2: mask2, 3: mask3}


def leaf_stats(X, y): # generiert von ChatGPT (Stand: 2025-12)
    masks = get_leaf_masks(X)
    classes = np.unique(y)

    stats = {}

    for leaf_id, m in masks.items():
        y_leaf = y[m]
        n = len(y_leaf)

        if n == 0:
            probs = {c: 0.0 for c in classes}
            maj = None
        else:
            probs = {c: (y_leaf == c).mean() for c in classes}
            maj = max(probs, key=probs.get)

        stats[leaf_id] = {
            "count": n,
            "probs": probs,
            "majority_class": maj
        }

    return stats, classes


# =========================
# 3. Baum-Plot mit Graphviz
# =========================
def plot_manual_tree(X, y): # generiert von ChatGPT (Stand: 2025-12)
    stats, classes = leaf_stats(X, y)

    # Farben für Klassen
    colors = {
        "Adelie":    "#8dd3c7",
        "Gentoo":    "#ffffb3",
        "Chinstrap": "#fb8072"
    }

    dot = Digraph()
    dot.attr(rankdir="TB")  # Top-Bottom-Baum

    # Innere Knoten (genau deine Bedingungen)
    dot.node("root",        "bill_length_mm <= 43.25?", shape='rect')
    dot.node("left_depth",  "bill_depth_mm <= 14.8?",   shape='rect')
    dot.node("right_depth", "bill_depth_mm <= 16.45?",  shape='rect')

    # Blätter: 0, 1, 2, 3
    for leaf_id in [0, 1, 2, 3]:
        info = stats[leaf_id]
        n = info["count"]
        probs = info["probs"]
        maj = info["majority_class"]

        if maj is None:
            fillcolor = "#ffffff"
            cls_label = "empty"
        else:
            fillcolor = colors.get(maj, "#ffffff")
            cls_label = maj

        prob_lines = "\n".join(
            [f"p({c}) = {probs[c]:.2f}" for c in classes]
        )
        label = f"{cls_label}\ncount = {n}\n{prob_lines}"

        dot.node(
            f"leaf{leaf_id}",
            label=label,
            style="filled",
            fillcolor=fillcolor,
            shape='box3d'
        )

    # Kanten: True immer "links" im logischen Sinne
    # Root: bill_length_mm <= 43.25?
    dot.edge("root", "left_depth",  label="True")
    dot.edge("root", "right_depth", label="False")

    # Linker Split: bill_depth_mm <= 14.8?
    dot.edge("left_depth", "leaf0", label="True")
    dot.edge("left_depth", "leaf1", label="False")

    # Rechter Split: bill_depth_mm <= 16.45?
    dot.edge("right_depth", "leaf2", label="True")
    dot.edge("right_depth", "leaf3", label="False")

    return dot

def make_grid(X, padding=1.0, n_grid=400): # generiert von ChatGPT (Stand: 2025-12)
    """
    Erzeugt xx, yy und grid_df aus einem 2D-Feature-DataFrame X.
    """
    f1, f2 = X.columns[0], X.columns[1]
    x_min, x_max = X[f1].min() - padding, X[f1].max() + padding
    y_min, y_max = X[f2].min() - padding, X[f2].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid)
    )

    grid_df = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=[f1, f2]
    )
    return xx, yy, grid_df, f1, f2
def plot_decision_boundary(
    X_train_2d,
    y_train_2d,
    *,
    model=None,
    model_name=None,
    X_test_2d=None,
    y_test_2d=None,
    which_points="train",
    use_train_test_range=False,
    manual=False,
    manual_rules=None,
    get_leaf_masks=None,
    ax=None,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    padding=1.0,
    n_grid=400,
    class_order=None,   
    palette=None       
): # generiert von ChatGPT (Stand: 2025-12)
    # 1) Punkte wählen
    if which_points == "test":
        if X_test_2d is None or y_test_2d is None:
            raise ValueError("Für which_points='test' müssen X_test_2d und y_test_2d übergeben werden.")
        X_points = X_test_2d
        y_points = y_test_2d
    else:
        X_points = X_train_2d
        y_points = y_train_2d

    # 2) Bounds
    if use_train_test_range and (X_test_2d is not None):
        X_for_bounds = pd.concat([X_train_2d, X_test_2d])
    else:
        X_for_bounds = X_train_2d

    # 3) Grid
    xx, yy, grid_df, f1, f2 = make_grid(X_for_bounds, padding=padding, n_grid=n_grid)

    # 4) Klassen-Reihenfolge + Palette (fix)
    if class_order is None:
        class_order = list(pd.unique(y_train_2d))  # stabiler als np.unique, aber immer noch "datenabhängig"

    if palette is None:
        # fallback: tab10, aber als dict gemappt auf class_order
        palette = dict(zip(class_order, sns.color_palette("tab10", len(class_order))))

    # Safety: prüfen, ob alle Labels im Mapping drin sind
    missing = set(pd.unique(pd.Series(y_train_2d))).difference(palette.keys())
    if missing:
        raise ValueError(f"Palette fehlt für Klassen: {missing}")

    # 5) Hintergrund: proba_max + boundaries
    if not manual:
        if model is None:
            raise ValueError("Für manual=False muss ein sklearn-Modell übergeben werden (model).")

        try:
            _ = model.classes_
        except AttributeError:
            model.fit(X_train_2d, y_train_2d)

        proba = model.predict_proba(grid_df)
        proba_max = proba.max(axis=1).reshape(xx.shape)

        Z_labels = model.predict(grid_df)
        Z_cat = pd.Categorical(Z_labels, categories=class_order, ordered=True)
        if (Z_cat.codes == -1).any():
            bad = set(pd.unique(Z_labels)) - set(class_order)
            raise ValueError(f"Vorhersagen enthalten unbekannte Klassen (nicht in class_order): {bad}")
        Z_numeric = Z_cat.codes.reshape(xx.shape)

    else:
        if manual_rules is None or get_leaf_masks is None:
            raise ValueError("Für manual=True müssen manual_rules und get_leaf_masks übergeben werden.")

        masks_train = get_leaf_masks(X_train_2d)
        stats = {}
        for leaf_id, m in masks_train.items():
            y_leaf = y_train_2d[m]
            n = len(y_leaf)
            if n == 0:
                probs = {c: 0.0 for c in class_order}
                maj = None
            else:
                probs = {c: (y_leaf == c).mean() for c in class_order}
                maj = max(probs, key=probs.get)
            stats[leaf_id] = {"count": n, "probs": probs, "majority_class": maj}

        leaf_confidence = {}
        for leaf_id, info in stats.items():
            maj = info["majority_class"]
            leaf_confidence[leaf_id] = 0.0 if maj is None else info["probs"][maj]

        masks_grid = get_leaf_masks(grid_df)
        leaf_id_grid = np.empty(len(grid_df), dtype=int)
        for leaf_id, m in masks_grid.items():
            leaf_id_grid[m.values] = leaf_id

        proba_max = np.array([leaf_confidence[l] for l in leaf_id_grid]).reshape(xx.shape)

        Z_labels = grid_df.apply(manual_rules, axis=1).values
        Z_cat = pd.Categorical(Z_labels, categories=class_order, ordered=True)
        if (Z_cat.codes == -1).any():
            bad = set(pd.unique(Z_labels)) - set(class_order)
            raise ValueError(f"Manual rules erzeugen unbekannte Klassen (nicht in class_order): {bad}")
        Z_numeric = Z_cat.codes.reshape(xx.shape)

    # 6) Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cont = ax.contourf(xx, yy, proba_max, alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(xx, yy, Z_numeric, levels=np.unique(Z_numeric), colors="black", linewidths=1)

    sns.scatterplot(
        x=X_points.iloc[:, 0],
        y=X_points.iloc[:, 1],
        hue=y_points,
        hue_order=class_order,   # <-- wichtig für stabile Legend-Reihenfolge
        palette=palette,         # <-- wichtig für stabile Farben
        edgecolor="black",
        s=60,
        ax=ax
    )

    if model_name is None:
        model_name = "Modell"

    punkt_text = "Train" if which_points == "train" else "Test"
    ax.set_title(f"{model_name} – {punkt_text}daten")
    ax.set_xlabel(X_points.columns[0])
    ax.set_ylabel(X_points.columns[1])
    ax.legend(title="Species", loc="lower right")

    return cont, ax

def bootstrapping(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    B: int = 1000,
    random_state: int = 42,
    metrics=("accuracy", "f1_macro"),
    min_oob: int = 10,
    alpha: float = 0.05
):

    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)
    df = X.copy()
    df["_y"] = y.values
    all_idx = df.index

    results = {m: [] for m in metrics}

    for b in range(B):
        train = df.sample(
            n=len(df),
            replace=True,
            random_state=random_state + b
        )

        oob_idx = all_idx.difference(train.index.unique())

        if len(oob_idx) < min_oob:
            continue

        test = df.loc[oob_idx]

        X_train = train.drop(columns="_y")
        y_train = train["_y"]
        X_test  = test.drop(columns="_y")
        y_test  = test["_y"]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if "accuracy" in metrics:
            results["accuracy"].append(accuracy_score(y_test, pred))
        if "f1_macro" in metrics:
            results["f1_macro"].append(f1_score(y_test, pred, average="macro"))

    summary = {}
    for m, vals in results.items():
        vals = np.asarray(vals, dtype=float)
        summary[m] = {
            "mean": float(vals.mean()),
            "ci": (float(np.percentile(vals, lower_p)), float(np.percentile(vals, upper_p))),
            "alpha": alpha,
            "n_effective": int(len(vals)),
            "distribution": vals,
    }
    return summary