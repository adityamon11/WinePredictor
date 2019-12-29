import pandas as pd;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score;

def WinePredict():
	nearest_neighbors = 11;
	data = pd.read_csv("WinePredictor.csv");
	print(data.head());
	print("Size of dataset ::",len(data));
	features = data.drop("Class",axis=1)
	print("Features are:::",features)
	labels = data["Class"];
	print("Labels are::",labels)

	classifier = KNeighborsClassifier(n_neighbors=nearest_neighbors);
	data_train,data_test,target_train,target_test=train_test_split(features,labels,test_size=0.25);
	classifier.fit(data_train,target_train);
	predictions = classifier.predict(data_test);
	accuracy = accuracy_score(target_test,predictions);
	return accuracy;

def main():
	print("------Wine Predictor using K nearest neighbour algorithm------");
	accuracy = WinePredict();
	print("Accuracy is::",accuracy*100,"%");


if __name__ == '__main__':
	main()