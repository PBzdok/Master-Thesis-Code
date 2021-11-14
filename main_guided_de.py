import datetime

import numpy as np
import streamlit as st
import torchxrayvision as xrv

from data import load_rsna_dataset, load_detailed_rsna_class_info, load_cluster_metadata, calculate_rsna_metrics

st.set_page_config(layout='wide')

model_specifier = 'densenet121-res224-rsna'
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')
d_rsna = load_rsna_dataset()

detailed_class_info = load_detailed_rsna_class_info()
classes = detailed_class_info['class'].unique()

cluster_metadata = load_cluster_metadata()

dataset = d_rsna.csv.merge(detailed_class_info[['patientId', 'class']], on='patientId')
dataset = dataset.merge(cluster_metadata[['anomaly_score', 'cluster', 'patientId']], on='patientId')

df_predictions, metrics = calculate_rsna_metrics(model, d_rsna)

st.title('KI Assessment System')

intentions = ["Allgemeine Informationen zur KI",
              "Standard-Metriken und Performanz der KI",
              "Trainingsdaten der KI",
              "Schwächen und Randfälle der KI",
              "Vergleich zwischen der KI und sich selbst"]

selection = st.radio("Mit welchem Thema möchten Sie sich befassen?", intentions)

if selection == intentions[0]:
    with st.expander('Übersicht'):
        st.subheader(f'{model_specifier}'.upper())
        overview_l, overview_r = st.columns(2)
        overview_l.text(f'Import Datum: {datetime.date.today()}')
        overview_r.text('Framework: Pytorch')

    with st.expander('Modellbeschreibung'):
        st.subheader('Modellbeschreibung')
        st.write(
            'Das KI-Modell "DENSENET121-RES224-RSNA" ist eine Variante der Architektur eines dicht vernetzten Faltungsnetzes (Densely Connected Convolutional Network). '
            'Es hat sich gezeigt, dass diese Architektur die beste Vorhersageleistung für Röntgenklassifizierungsaufgaben erbringt. '
            'Das Modell wurde mit großen Mengen öffentlich zugänglicher Röntgendaten trainiert und vom Labor für maschinelles Lernen und Medizin (mlmed.org) bereitgestellt. '
            'Das Modell erwartet eine Eingabe von 224x224 Pixeln (Graustufen) und gibt die Wahrscheinlichkeit von 18 verschiedenen Pathologien aus, wobei diese Version auf Lungenentzündung spezialisiert ist und nur diese vorhersagt.'
        )

    with st.expander('Fähigkeiten'):
        st.subheader('Fähigkeiten')
        st.write('Das Modell ist in der Lage, folgende Pathologien anhand von Röntgenbildern vorherzusagen:')
        st.markdown('* Lungenentzündung')

elif selection == intentions[1]:
    with st.expander('Standard-Metriken'):
        st.subheader('Standard-Metriken')
        metrics1, metrics2, metrics3, metrics4 = st.columns(4)
        metrics1.metric(label='Vertrauenswahrscheinlichkeit', value=str(metrics['accuracy'].round(2)))
        metrics2.metric(label='Genauigkeit', value=str(metrics['precision'].round(2)))
        metrics3.metric(label='Sensitivität', value=str(metrics['recall'].round(2)))
        metrics4.metric(label='F1-Maß', value=str(metrics['f1'].round(2)))

elif selection == intentions[2]:
    def set_browse_indices(high):
        st.session_state.indices = np.random.randint(low=0, high=high, size=10)


    with st.expander('Daten durchsuchen'):
        st.subheader('Daten durchsuchen')
        n_pathological = sum(dataset['Target'] == 1)
        n_non_pathological = sum(dataset['Target'] == 0)
        st.write(
            f'Dieser Datensatz enthält {n_pathological} pathologische Bilder und {n_non_pathological} nicht-pathologische Bilder')

        rows = list()
        chunk_size = 3
        options = []

        for i in range(0, len(classes), chunk_size):
            rows.append(classes[i:i + chunk_size])

        for row in rows:
            labels_columns = st.columns(len(row))
            for _, (label, column) in enumerate(zip(row, labels_columns)):
                if column.checkbox(label, value=True, key=label):
                    options.append(label)

        if 'indices' not in st.session_state:
            set_browse_indices(len(dataset.index))
        index_list = st.session_state.indices

        df_samples = dataset.loc[index_list]
        df_samples = df_samples[df_samples['class'].isin(options)]

        selected_idx = st.selectbox('Zeile wählen:', df_samples.index.values)
        patient_id = df_samples['patientid'][selected_idx]
        image_id = patient_id

        df_samples = df_samples[['class', 'Target']]
        df_samples = df_samples.rename(columns={'class': 'Details', 'Target': 'Nachweis von Lungenentzündung'})

        left_column, right_column = st.columns(2)
        left_column.table(df_samples)
        if left_column.checkbox('Metadaten anzeigen'):
            st.table(dataset.loc[dataset['patientid'] == patient_id][
                         ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined', 'ConversionType']])

        left_column.button('Mehr Bilder anzeigen', on_click=set_browse_indices, args=(len(dataset.index),))

        image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{image_id}.jpg'
        right_column.image(image_path, caption=f'{image_id}')


    # -----------------------------------------------------------------------------------------------------------

    def set_cluster_index(index, cluster):
        st.session_state.cluster_index = index
        set_cluster_sample_index(cluster.index)


    def set_cluster_sample_index(indices):
        st.session_state.cluster_sample_index = np.random.choice(indices, 1)


    with st.expander('Datengruppen'):
        st.subheader('Datengruppen')

        cluster_labels = dataset['cluster'].unique()
        cluster_labels = np.sort(cluster_labels)
        n_clusters = len(cluster_labels)

        clusters = dataset.groupby('cluster')

        st.write(
            f'Eine Datensatzanalyse hat gezeigt, dass {n_clusters} Hauptgruppen von Daten durch Metadaten aggregiert werden können.')

        left_cluster_column, mid_cluster_column, right_cluster_column = st.columns([0.5, 2, 3])

        if 'cluster_index' not in st.session_state:
            set_cluster_index(0, clusters.get_group(0))
        cluster_index = st.session_state.cluster_index
        selected_cluster = clusters.get_group(cluster_index)

        for c in cluster_labels:
            left_cluster_column.button(f'Gruppe {c}', key=f'Cluster {c}', on_click=set_cluster_index,
                                       args=(c, clusters.get_group(c),))

        mean_anomaly_score = selected_cluster['anomaly_score'].mean()
        mean_age = selected_cluster['PatientAge'].mean()
        max_age = selected_cluster['PatientAge'].max()
        min_age = selected_cluster['PatientAge'].min()
        n_men = sum(selected_cluster['PatientSex'] == 'M')
        n_women = sum(selected_cluster['PatientSex'] == 'F')
        n_pathological_cluster = sum(selected_cluster['Target'] == 1)
        n_non_pathological_cluster = sum(selected_cluster['Target'] == 0)

        mid_cluster_column.markdown(
            f'* Gruppe {cluster_index} hat {len(selected_cluster.index)} Einträge und einen mittleren Anomalie-Score von {mean_anomaly_score.round(2)}.\n'
            f'* Das mittlere Alter der Gruppe ist {mean_age.round(2)}, während das maximale bzw. minimale Alter {max_age} und {min_age} sind.\n'
            f'* Die Gruppe enthält {n_men} männliche Fälle und {n_women} weibliche Fälle.\n'
            f'* Die Gruppe enthält {n_pathological_cluster} pathologische Fälle und {n_non_pathological_cluster} nicht pathologische Fälle.\n'
        )

        cluster_sample_index = st.session_state.cluster_sample_index
        cluster_sample_id = selected_cluster['patientid'][cluster_sample_index].values[0]
        image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{cluster_sample_id}.jpg'
        right_cluster_column.image(image_path)
        right_cluster_column.table(dataset.loc[dataset['patientid'] == cluster_sample_id][
                                       ['PatientAge', 'PatientSex', 'Target', 'anomaly_score']])
        right_cluster_column.button(f'Weitere Beispiele aus Gruppe {cluster_index} zeigen',
                                    on_click=set_cluster_sample_index, args=(selected_cluster.index,))

elif selection == intentions[3]:
    def set_fp_indices(indices):
        st.session_state.fp_indices = np.random.choice(indices, 10)


    def set_fn_indices(indices):
        st.session_state.fn_indices = np.random.choice(indices, 10)


    with st.expander('Grenzen des Modells'):
        st.subheader('Grenzen des Modells')

        df_limit_samples = df_predictions
        fp_index_list = []
        fn_index_list = []

        case = st.radio(
            'Wählen Sie einen Randfall:',
            ['Falsch Positiv', 'Falsch Negativ']
        )
        if case == 'Falsch Positiv':
            df_limit_samples = df_limit_samples[(df_limit_samples['y_true'] == 0) & (df_limit_samples['y_pred'] == 1)]
            if 'fp_indices' not in st.session_state:
                set_fp_indices(df_limit_samples.index.values)
            fp_index_list = st.session_state.fp_indices
            df_limit_samples = df_limit_samples.loc[fp_index_list]
        else:
            df_limit_samples = df_limit_samples[(df_limit_samples['y_true'] == 1) & (df_limit_samples['y_pred'] == 0)]
            if 'fn_indices' not in st.session_state:
                set_fn_indices(df_limit_samples.index.values)
            fn_index_list = st.session_state.fn_indices
            df_limit_samples = df_limit_samples.loc[fn_index_list]

        selected_limit_idx = st.selectbox('Zeile wählen:', df_limit_samples.index)
        patient_limit_id = df_limit_samples.drop_duplicates()['patientid'][selected_limit_idx]
        image_limit_id = patient_limit_id

        df_limit_samples = df_limit_samples[['y_pred', 'y_true']].astype(int)

        df_limit_samples = df_limit_samples.rename(
            columns={'y_pred': 'Vorhersage', 'y_true': 'Nachweis von Lungenentzündung'})

        left_limit_column, right_limit_column = st.columns(2)
        right_limit_column.table(df_limit_samples)
        if right_limit_column.checkbox('Zeige Metadaten', key='limits'):
            st.table(dataset.loc[dataset['patientid'] == patient_limit_id][
                         ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined', 'ConversionType']])

        image_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{image_limit_id}.jpg'
        left_limit_column.image(image_path, caption=f'{image_limit_id}')

        if left_limit_column.checkbox('Erklärung anzeigen, basierend auf welchen Bildbereichen die KI entschieden hat'):
            st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{image_limit_id}.jpg',
                     caption="Helle / gelbe Bereiche zeigen Bereiche mit hoher Bedeutung an!")

elif selection == intentions[4]:
    def set_experiment_index(high):
        st.session_state.index = np.random.randint(low=0, high=high, size=1)


    def show_occlusion(id):
        st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{id}.jpg',
                 caption="Helle / gelbe Bereiche zeigen Bereiche mit hoher Bedeutung an!")


    with st.expander('Experiment'):
        st.subheader('Experiment')

        if 'index' not in st.session_state:
            set_experiment_index(len(df_predictions.index))
        rnd_idx = st.session_state.index

        sample = df_predictions.iloc[rnd_idx[0]]
        sample_id = sample['patientid']
        sample_pred = sample['y_pred']

        experiment_sample_path = f'./data/kaggle-pneumonia-jpg/stage_2_train_images_jpg/{sample_id}.jpg'

        experiment_left_column, experiment_right_column = st.columns(2)
        experiment_left_column.image(experiment_sample_path, caption=f'{sample_id}.jpg')

        prediction = experiment_right_column.radio('Glauben Sie, dass das Bild pathologisch ist?', ('Ja', 'Nein'))

        if experiment_right_column.button('Zeige Ergebnis'):
            if (prediction == 'Ja' and sample_pred == 1.0) or (prediction == 'Nein' and sample_pred == 0.0):
                experiment_right_column.success('Sie und die KI haben die gleiche Meinung!')
            else:
                experiment_right_column.error('Sie und die KI haben unterschiedliche Meinungen!')

        experiment_right_column.button('Nochmal versuchen', on_click=set_experiment_index,
                                       args=(len(df_predictions.index),))

        if experiment_right_column.checkbox('Zeige Metadaten', key='experiment'):
            experiment_right_column.table(dataset.loc[dataset['patientId'] == sample_id][
                                              ['PatientAge', 'PatientSex', 'ViewPosition', 'BodyPartExamined']])
        if experiment_right_column.checkbox(
                'Erklärung anzeigen, basierend auf welchen Bildbereichen die KI entschieden hat',
                key='experiment'):
            st.image(f'./data/kaggle-pneumonia-jpg/occlusion/{sample_id}.jpg',
                     caption="Helle / gelbe Bereiche zeigen Bereiche mit hoher Bedeutung an!")
