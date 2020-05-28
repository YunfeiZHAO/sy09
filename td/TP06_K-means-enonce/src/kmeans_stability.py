def kmeans_dataset(n_clusters_list, strategies, tries):
    def run():
        for n_clusters in n_clusters_list:
            for strategy in strategies:
                for rs in range(tries):  # On utilisera `rs` pour fixer le `random_state`
                    km = ...
                    inertia = ...
                    yield rs, strategy, n_clusters, inertia

    return run


gen = kmeans_dataset(...)

df = pd.DataFrame(gen(), columns=["seed", "init", "n_clusters", "inertia"])
df = df.astype({
    "seed": "int32",
    "n_clusters": "int32"
})
