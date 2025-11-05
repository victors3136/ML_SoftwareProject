from lib.OrdinalEncoder import OrdinalEncoder

if __name__ == "__main__":
    e = OrdinalEncoder()
    print(e.fit(["1", "2", "3", "4", "1", "2", "3"]))
