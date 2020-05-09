#======== DATA USPS DIGITS - Model and settings configuration

# model instantiation -----------------------------------------------

library(keras)

FLAGS <- flags(
  flag_integer("hdlayer_1", 256),
  flag_numeric("dropout_1", 0.2),
  flag_integer("hdlayer_2", 64),
  flag_numeric("dropout_2", 0.1),
  flag_integer("hdlayer_3", 16),
  flag_numeric("dropout_3", 0.05)
)

# model configuration
model <- keras_model_sequential() %>%
  layer_dense(units =  FLAGS$hdlayer_1, input_shape = ncol(x1_train), activation = "relu", name = "layer_1",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = FLAGS$dropout_1) %>%
  layer_dense(units = FLAGS$hdlayer_2, activation = "relu", name = "layer_2",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = FLAGS$dropout_2) %>%
  layer_dense(units = FLAGS$hdlayer_3, activation = "relu", name = "layer_3",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = FLAGS$dropout_3) %>%
  layer_dense(units = ncol(yc_train), activation = "softmax", name = "layer_out") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_adam(lr = 0.001),
  )

fit <- model %>% fit(
  x = x1_train, y = yc_train,
  validation_data = list(x1_val, yc_val),
  epochs = 100,
  batch_size = 64,
  verbose = 0,
  callbacks = callback_early_stopping(monitor = "val_accuracy", patience = 20)
)

# store accuracy on test set for each run
score <- model %>% evaluate(
  x1_test, yc_test,
  verbose = 0
)