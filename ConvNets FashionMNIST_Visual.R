library(knitr)
opts_chunk$set(message = FALSE, warning = FALSE, cache = TRUE, cache.lazy = FALSE)
options(width = 120, dplyr.width = 120)
library(ggplot2)
theme_set(theme_light())


library(readr)
library(keras)


# Data downloaded from https://www.kaggle.com/zalando-research/fashionmnist

TrainSetData <- read_csv("fashion-mnist_train.csv",
                       col_types = cols(.default = "i"))
TestSetData <- read_csv("fashion-mnist_test.csv",
                      col_types = cols(.default = "i"))

# Fashion MNIST Image Data is 28*28 pixels
ImgRows <- 28
ImgCols <- 28

# Data Preparation

Train_X <- as.matrix(TrainSetData[, 2:dim(TrainSetData)[2]])
Train_Y <- as.matrix(TrainSetData[, 1])

# Unflattening the data.
dim(Train_X) <- c(nrow(Train_X), ImgRows, ImgCols, 1) 

Test_X <- as.matrix(TestSetData[, 2:dim(TrainSetData)[2]])
Test_Y <- as.matrix(TestSetData[, 1])
dim(Test_X) <- c(nrow(Test_X), ImgRows, ImgCols, 1) 


Fashion_Labels<-c( "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# Function to rotate matrices
rotate <- function(x) t(apply(x, 2, rev))

# Function to plot image from a matrix x
plot_image <- function(x, title = "", title.color = "black") {
  dim(x) <- c(ImgRows, ImgCols)
  image(rotate(rotate(x)), axes = FALSE,
        col = grey(seq(0, 1, length = 256)),
        main = list(title, col = title.color))
}

# Plot images from the training set
par(mfrow=c(4, 4), mar=c(0, 0.2, 1, 0.2))
for (i in 1:16) {
  n_row <- i * 10
  plot_image(Train_X[n_row, , , 1],
            Fashion_Labels[as.numeric(TrainSetData[n_row, 1] + 1)])
}

# Hyperparameters Setting
batch_size <- 256
num_classes <- 10
epochs <- 40

input_shape <- c(ImgRows, ImgCols, 1)

Train_X <- Train_X / 255
Test_X <- Test_X / 255


Train_Y <- to_categorical(Train_Y, num_classes)
Test_Y <- to_categorical(Test_Y, num_classes)

# Convolutional Nerual Network Model 

model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu',
                input_shape = input_shape) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = 'softmax')

summary(model)
# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# train and evaluate
model %>% fit(
  Train_X, Train_Y,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(Test_X, Test_Y)
)

# Saving the model for visulization

model %>% save_model_hdf5("CNN_Fashion.h5")
summary(model)

# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# Visualization the activation of Convolutional layers 
plot(as.raster(Train_X[1002,,,1]))
img_tensor <- Train_X[1002,,,1]
img_tensor <- array_reshape(img_tensor, c(1, 28, 28, 1))

dim(img_tensor)

activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = terrain.colors(12))
}

for (i in 1:8) {
  plot_channel(first_layer_activation[1,,,i])
}


dir.create("Fashion_activations")
image_size <- 58
images_per_row <- 16

for (i in 1:8) {
 
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("Fashion_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}
#Evaluation the model


Model_Performance <- model %>% evaluate(
  Test_X, Test_Y, verbose = 0
)
cat('Test loss:', Model_Performance[[1]], '\n')
cat('Test accuracy:', Model_Performance[[2]], '\n')


# Visulization the model predictions
for (i in 1:32) {
  n_row <- i * 10
  T_Tensor <- Train_X[n_row, , , 1]
  dim(T_Tensor) <- c(1, ImgRows, ImgCols, 1)
  pred <- model %>% predict(T_Tensor)
  plot_image(Train_X[n_row, , , 1],
             Fashion_Labels[which.max(pred)],
             "red")
}

