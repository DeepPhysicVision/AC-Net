import h5py
from ExampleReader import ExampleReader
from Evaluator import Evaluator
from keras.models import Model
import numpy as np
import model as m

if __name__ == '__main__':
    model_path = 'result/m_'

    example_reader = ExampleReader()
    position_matrix = example_reader.load_position_matrix()

    train_aspect_labels, train_aspect_text_inputs, train_sentence_inputs, _ = example_reader.load_inputs_and_label(name='train')
    test_aspect_labels, test_aspect_text_inputs, test_sentence_inputs, test_true_labels = example_reader.load_inputs_and_label(name='test')

    train_sentence_inputs, train_aspect_text_inputs, train_positions, _ = example_reader.get_position_input(train_sentence_inputs,train_aspect_text_inputs)
    test_sentence_inputs, test_aspect_text_inputs, test_positions, _ = example_reader.get_position_input(test_sentence_inputs,test_aspect_text_inputs)

    embedding_matrix = example_reader.get_embedding_matrix()
    position_ids = example_reader.get_position_ids(max_len=78)
    example_reader.convert_position(position_inputs=train_positions, position_ids=position_ids)
    example_reader.convert_position(position_inputs=test_positions, position_ids=position_ids)

    train_aspects = example_reader.pad_aspect_index(train_aspect_text_inputs.tolist(), max_length=22)
    test_aspects = example_reader.pad_aspect_index(test_aspect_text_inputs.tolist(), max_length=22)

    model = m.build_model(max_len=78,
                          aspect_max_len=22,
                          embedding_matrix=embedding_matrix,
                          position_embedding_matrix=position_matrix,
                          num_words=5144)

    evaluator = Evaluator(true_labels=test_true_labels, sentences=test_sentence_inputs, aspects=test_aspect_text_inputs)

    epoch = 1
    while epoch <= 50:
        model = m.train_model(sentence_inputs=train_sentence_inputs,
                              position_inputs= train_positions,
                              aspect_input=train_aspects,
                              labels=train_aspect_labels,
                              model=model)
        results = m.get_predict(sentence_inputs=test_sentence_inputs,
                            position_inputs=test_positions,
                            aspect_input=test_aspects,
                            model=model)
        print("\n--epoch"+str(epoch)+"--")
        F, acc = evaluator.get_macro_f1(predictions=results, epoch=epoch)
        if epoch % 2 == 0:
            print("current max f1 score"+str(evaluator.max_F1))
            print("max f1 is gained in epoch"+str(evaluator.max_F1_epoch))
            print("current max acc"+str(evaluator.max_acc))
            print("max acc is gained in epoch"+str(evaluator.max_acc_epoch))
        print("happy ending")

        if acc > 0.8000:
            model.save_weights(model_path+"_acc_"+str(acc*100)+"_F_"+str(F*100)+"_"+str(epoch))
        elif F > 0.7100:
            model.save_weights(model_path + "_acc_" + str(acc * 100) + "_F_" + str(F * 100) + "_" + str(epoch))
        epoch += 1


