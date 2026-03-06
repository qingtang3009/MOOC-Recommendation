1. Autoencoder is used to acquire features of users and resources from text data, it transfers logical user data (personal information)
and resource data (course introductions) into low-dimensional feature embeddings (user embeddings and resource embeddings), the dimension is 8.

2. TransD is used to acquire features of users and resources from a knowledge graph (this knowledge graph was built according to
the history of users, mainly refer to the users' selection records), it transfers structure data (users and resources are intricately
linked in the knowledge graph) into low-dimensional feature embeddings (user embeddings and resource embeddings), the dimension is 128.

3. The feature embeddings (combine the feature embeddings from different sources respectively, the dimension is 8+128) are used to train the MLP model and make
predictions (recomendations) for the new users.
