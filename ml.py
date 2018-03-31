from sklearn import tree

# 1 kolumna, moc silnika
# 2 kolumna, liczba pasazerow
# 3 kolumna: waga
features = [
    [70, 5, 1100],
    [113, 5, 1300],
    [95, 5, 1230],
    [605, 2, 1450],
    [530, 2, 1650],
    [789, 2, 1050],
    [240, 12, 1900],
    [267, 18, 2000],
    [190, 10, 1819]
]
# 0 - osobowka, 1 - sportowy, 2 - bus
labels = [0,0,0,1,1,1,2,2,2]
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print(clf.predict([
    [120, 4, 1170],
    [450, 2, 1470]
]))