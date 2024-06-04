# Machine Learning
# Installer les packages nécessaires (si ce n'est pas déjà fait)
install.packages("caret")
install.packages("randomForest")
install.packages("ggplot2")
install.packages("e1071")

# Charger les bibliothèques
library(caret)
library(randomForest)
library(ggplot2)

# Charger les données Iris
data(iris)
# Séparation des données en ensembles d'entraînement (80%) et de test (20%)
set.seed(42)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]
# Entraînement du modèle Random Forest
set.seed(42)
model <- randomForest(Species ~ ., data = trainData, ntree = 100)
# Prédiction sur l'ensemble de test
predictions <- predict(model, testData)
# Création de la matrice de confusion
confusionMatrix <- confusionMatrix(predictions, testData$Species)
print(confusionMatrix)
# Conversion de la matrice de confusion en DataFrame pour ggplot
confMatrix <- as.data.frame(confusionMatrix$table)
names(confMatrix) <- c("Reference", "Prediction", "Freq")


# Visualisation avec ggplot2
ggplot(data = confMatrix, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matrice de Confusion", x = "Vraie Étiquette", y = "Prédiction") +
  theme_minimal()

                
                