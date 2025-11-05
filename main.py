from lib.OrdinalEncoder import OrdinalEncoder

if __name__ == "__main__":
    e = OrdinalEncoder()
    print(e.fit(["man", "woman", "child", "elder", "man", "woman", "child"]))
