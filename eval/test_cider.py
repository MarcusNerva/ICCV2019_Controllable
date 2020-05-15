from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from collections import OrderedDict

def test_sentences(sent1, sent2):
    predictions = OrderedDict()
    references = OrderedDict()

    predictions[0] = [sent1]
    references[0] = [sent2, sent2]

    predictions = {0: predictions[0]}
    references = {0: references[0]}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)


if __name__ == '__main__':
    test_sentences('a boy is playing with a girl', 'a girl is playing with a boy')
    test_sentences('a boy is playing with a girl', 'a boy is playing with a women')
