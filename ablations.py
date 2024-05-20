# %%
import pandas as pd

# %%
df = pd.read_csv("ablation_results.csv")
df2 = pd.read_csv("ablation_results_tied.csv")
# merge dfs
df = pd.concat([df, df2])
for col in ["phone.oem", "phone.model", "phone.network_edge"]:
  df[col] = 1 - df[col]

# %%
def make_nice_latex(df, groupby_things, selection=None):
    df = df.copy()
    if selection is not None:
        df = df[selection]

    idxmin = df.groupby(groupby_things)[[x for x in df.columns if "phone." in x]].agg("mean").transpose().idxmin(axis=1)

    df = df.groupby(groupby_things)[[x for x in df.columns if "phone." in x]].agg(
        ["mean", "std"]
    )
    # find the indices for each row where the mean is smallest

    def make_str_lambda(column):
      def make_str(x):
        for k,v in idxmin.to_dict().items():
          if v == x.name:
            if k == column:
              return f"$\mathbf{{{x[(column, 'mean')]:.3f}_{{\pm {x[(column, 'std')]:.3f}}}}}$"
        return f"${x[(column, 'mean')]:.3f}_{{\pm {x[(column, 'std')]:.3f}}}$"
      return make_str

    for column in set(col[0] for col in df.columns):
        # Create a new column with formatted string
        df[(column, "combined")] = df.apply(
            make_str_lambda(column), axis=1
        )
    df = df.drop(columns=[col for col in df.columns if col[1] in ["mean", "std"]])
    features = [
        "weight",
        "height",
        "depth",
        "width",
        "display_size",
        "battery",
        "launch.day",
        "launch.month",
        "launch.year",
        "oem",
        "network_edge",
        "model",
    ]
    df.columns = [
        "".join(col).replace("phone.", "").replace("combined", "").strip()
        if isinstance(col, tuple)
        else col
        for col in df.columns
    ]
    df = df[features]
    df.columns = [x.replace("_", "-") for x in df.columns]
    replaces = {
                "train_mask_rate": "mask_rate",
                "d_model": "model dim",
                "num_emb": "num. emb. type",
                "num_decoder_mixtures": "GMM mixtures",
                "tie_numerical_embeddings": "num. emb. tied",
                "_": "-",
            }
    for k, v in replaces.items():
        df.index.names = [x.replace(k, v) for x in df.index.names]

    return df


# %%
# all ablations
groupby_things = [
    "num_emb",
    "d_model",
    "lr",
    "train_mask_rate",
    "num_decoder_mixtures",
    "tie_numerical_embeddings",
]
# sel = (df.num_decoder_mixtures == 50) & (df.lr == 0.001) & (df.d_model == 512)
print(make_nice_latex(df, groupby_things).to_latex())

# %%
groupby_things = ["num_emb", "tie_numerical_embeddings"]
sel = (df.num_decoder_mixtures == 50) & (df.lr == 0.001) & (df.d_model == 512) & (df.train_mask_rate == -1)
nicer = make_nice_latex(df, groupby_things, sel).transpose()
print(nicer.to_latex())
# %%

groupby_things = [
    "num_emb",
    # "d_model",
    # "lr",
    # "train_mask_rate",
    "num_decoder_mixtures",
    # "tie_numerical_embeddings",
]
sel = (df.lr == 0.001) & (df.d_model == 512) & (df.train_mask_rate == -1) & (df.tie_numerical_embeddings == 0)
nicer = make_nice_latex(df, groupby_things, sel).transpose()
print(nicer.to_latex())

# %%

groupby_things = [
    # "num_emb",
    # "d_model",
    # "lr",
    "tie_numerical_embeddings",
    "train_mask_rate",
    # "num_decoder_mixtures",
]
sel = (df.num_emb == "dice") & (df.d_model == 512) & (df.num_decoder_mixtures == 50) & (df.lr == .001)
nicer = make_nice_latex(df, groupby_things, sel).transpose()
print(nicer.to_latex())

# %%
