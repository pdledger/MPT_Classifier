def Plot_comparison_features(Frequencies,X_train_norm,Y_train):

        
    feature1_real = X_train_norm[:, 0:len(Frequencies)]
    feature1_imag = X_train_norm[:, len(Frequencies):2 * len(Frequencies)]

    feature2_real = X_train_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
    feature2_imag = X_train_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]

    feature3_real = X_train_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
    feature3_imag = X_train_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

    object_class = Y_train[:, None] @ np.ones((1, len(Frequencies)))
    object_omega = np.ones((X_train_norm.shape[0], 1)) @ Frequencies[None, :]

    internal_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                          'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                          'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                          'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

    internal_dataframe = pd.DataFrame(internal_data_dict)

    feature1_real = X_test_norm[:, 0:len(Frequencies)]
    feature1_imag = X_test_norm[:, len(Frequencies):2 * len(Frequencies)]

    feature2_real = X_test_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
    feature2_imag = X_test_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]

    feature3_real = X_test_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
    feature3_imag = X_test_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

    object_class = np.asarray(['External Data']* len(Frequencies))[:,None]
    object_omega = np.ones((X_test_norm.shape[0], 1)) @ Frequencies[None, :]

    external_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                          'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                          'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                          'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

    external_dataframe = pd.DataFrame(external_data_dict)
    total_dataframe = pd.concat([internal_dataframe, external_dataframe], ignore_index=True, sort=False)

    plt.figure()
    palette = sns.color_palette("Paired", 6)
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                     y='pri1_imag', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.figure()
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                     y='pri1_real', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.figure()
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                  y='pri2_real', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.figure()
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                  y='pri2_imag', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.figure()
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                  y='pri3_real', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.figure()
    total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                  y='pri3_imag', hue='class', palette=palette)
    total_lineplot.set(xscale='log')

    plt.show()
