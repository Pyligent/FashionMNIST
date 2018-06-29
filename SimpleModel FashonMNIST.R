library(knitr)
opts_chunk$set(message = FALSE, warning = FALSE, cache = TRUE, cache.lazy = FALSE)
options(width = 100, dplyr.width = 100)
library(ggplot2)
theme_set(theme_light())

# The fashionMNIST dataset also can be 
# accessed using the dataset_fashion_mnist() function in Keras.
library(keras)

# Data Preparation
FashionMNIST <- dataset_fashion_mnist()
Train_X <- FashionMNIST$train$x
Train_Y <- FashionMNIST$train$y
Test_X <- FashionMNIST$test$x
Test_Y <- FashionMNIST$test$y


# Reshape
Train_X <- array_reshape(Train_X, c(nrow(Train_X), 784)) 
Test_X <- array_reshape(Test_X, c(nrow(Test_X), 784))

# Value Normalization 
Train_X <- Train_X / 255
Test_X <- Test_X / 255


Train_Y <- to_categorical(Train_Y, 10)
Test_Y <- to_categorical(Test_Y, 10)


# The simple deep learning model

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# Setting the loss function, optimizar and regularization 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c('accuracy')
)

# Model Evaluation

history <- model %>% fit(
  Train_X, Train_Y, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
); plot(history)

# Evaluate the model’s performance on the test data:
model %>% evaluate(Test_X, Test_Y)

# Generate predictions on new data:
model %>% predict_classes(Test_X)
