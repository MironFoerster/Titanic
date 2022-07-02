import tensorflow as tf
tf.config.run_functions_eagerly(True)

def remove_features(x, y):
    return {"age":x["age"], "embarked":x["embarked"], "fare":x["fare"], "parch":x["parch"], "pclass":x["pclass"], "sex":x["sex"], "sibsp":x["sibsp"]}, y


def preprocess_features(x, y):
    # supported features: age(norm), embarked(onehot), fare(norm), parch(norm), pclass(onehot), sex(leave), sibsp(norm)
    maxs = {"age": 100, "fare": 600, "parch": 10, "sibsp": 10}
    fills = {"age": 28., "embarked": 2, "fare": 20., "parch": 0, "pclass": 3, "sex": 0, "sibsp": 0}
    if x["age"] == -1.:
        x["age"] = tf.constant(fills["age"])
    if x["fare"] == -1.:
        x["fare"] = tf.constant(fills["fare"])

    x["emb"] = tf.one_hot(x["embarked"], 3, dtype=tf.float32)
    x["pc"] = tf.one_hot(x["pclass"], 3, dtype=tf.float32)

    y = tf.cast(y, tf.float32)
    x["sex"] = tf.cast(x["sex"], tf.float32)
    print(x["parch"]/maxs["parch"])

    print((x["age"]/maxs["age"], x["emb"][0], x["emb"][1], x["emb"][2], x["fare"]/maxs["fare"], tf.cast(x["parch"]/maxs["parch"], tf.float32), x["pc"][0], x["pc"][1], x["pc"][2], x["sex"], tf.cast(x["sibsp"]/maxs["sibsp"], tf.float32)), y)
    return (x["age"]/maxs["age"], x["emb"][0], x["emb"][1], x["emb"][2], x["fare"]/maxs["fare"], tf.cast(x["parch"]/maxs["parch"], tf.float32), x["pc"][0], x["pc"][1], x["pc"][2], x["sex"], tf.cast(x["sibsp"]/maxs["sibsp"], tf.float32)), y