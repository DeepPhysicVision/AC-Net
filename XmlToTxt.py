try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class xml_to_txt(object):
    def __init__(self, file=""):
        self.file = file
        self.dir = 'data_restaurant/'

    def convert_xml_to_txt(self, name=""):
        tree = ET.parse(self.file)
        root = tree.getroot()

        sen_num = 0
        aspect_num = 0
        sen_without_aspect_num = 0

        text_file = self.dir + name + "_text.txt"
        aspect_text_file = self.dir + name + "_aspects_text.txt"
        aspect_label_file = self.dir + name + "_aspects_label.txt"

        text_file = open(text_file, 'w')
        aspect_text_file = open(aspect_text_file, 'w')
        aspect_label_file = open(aspect_label_file, 'w')

        ### read xml file
        for sentence in root:
            sen_num += 1

            text = sentence.find("text")
            temp = text.text.strip()
            while '  ' in temp:
                temp = temp.replace('  ', ' ')
            temp = temp.replace(u'\xa0', ' ').lower()
            #print(temp)
            text_file.write(temp + "\n")

            aspect_terms = sentence.find("aspectTerms")
            if aspect_terms is not None:
                for aspect in aspect_terms:
                    aspect_num += 1
                    aspect_text_file.write(aspect.get("term").replace(u'\xa0', ' ').lower() + "\n")
                    aspect_label_file.write(aspect.get("polarity") + "#" + str(sen_num - 1) + "\n")
            else:
                sen_without_aspect_num += 1

        text_file.close()
        aspect_text_file.close()
        aspect_label_file.close()
        print("There are " + str(sen_num) + " sentences.")  #3041 800
        print("There are " + str(sen_without_aspect_num) + " sentences without aspects.") #1020 194
        print("There are " + str(aspect_num) + " aspects.")  #3693 1134

if __name__ == '__main__':
    train_corpus = 'data_restaurant/train.xml'
    test_corpus = 'data_restaurant/test.xml'

    train_pre = xml_to_txt(train_corpus)
    train_pre.convert_xml_to_txt(name="train")

    test_pre = xml_to_txt(test_corpus)
    test_pre.convert_xml_to_txt(name="test")
