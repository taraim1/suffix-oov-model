"""
이 파일은 data 폴더의 corpus.json 파일을 preprocessing합니다.
"""


import json
from collections import defaultdict
# from copy import deepcopy
from collections import Counter

S_total_count = defaultdict(int) # 접미 형태소열 S가 등장한 총 횟수입니다.
S_noun_count = defaultdict(int) # 접미 형태소열 S가 명사 뒤에 등장한 횟수입니다.
s_noun_map = defaultdict(set) # 특정 S가 어떤 N들과 결합했는지 저장합니다.
noun_s_map = defaultdict(set) # 특정 N이 어떤 S들과 결합했는지 저장합니다.
co_occurrence_frequencies = defaultdict(list) # 공기 빈도 (Key: S, Value: [(SB1, count), (SB2, count)...(SBk, count)], 이때 k는 논문을 따라 5로 정했습니다.)
K = 5

"""
S_total_count와 S_noun_count 두 개만 있으면 기본적인 P(N=1|S)는 구할 수 있지만, 논문에서 핵심적으로 다루는 P(U|N=1,S)는 구하기 어렵습니다.
이 논문은 말뭉치에서 등장하지 않은 U가 관심 대상이기 때문입니다.
즉 P(U|N=1,SB)를 구할 수 있게 해주는 적절한 지표 형태소열 SB를 구하는 게 문제 해결의 핵심입니다.
따라서 s_noun_map, noun_s_map을 이용해 공기 빈도(S와 SB가 얼마나 동시에 나오는가)를 계산하고, 임의의 S에 대해 적절한 SB들을 구할 수 있도록 했습니다.

"""

input_path = 'data/corpus.json'
output_path = 'data/model_data.json'

with open(input_path, 'r', encoding='utf8') as f:
    data = json.load(f)

    docs = data['document']

    for doc in docs:
        senteces = doc['sentence']

        for sentence in senteces:
            morphemes = sentence['morpheme']
            
            # 형태소들을 어절 단위로 묶어 리스트에 넣습니다.
            eojeols = []
            eojeol_count = 0
            
            for morpheme in morphemes:
                if morpheme['word_id'] != eojeol_count:
                    eojeol_count += 1
                    eojeols.append([])
                
                eojeols[eojeol_count-1].append({'form': morpheme['form'], 'label': morpheme['label']})

            for eojeol in eojeols:

                # 논문 내용에 의거, 명사열과 명사에 결합된 접두사 및 관형사를 NNG로 합칩니다.
                # 기본적으로 XPN + NNG/NNP, MM + NNG/NNP, NNG/NNP + NNG/NNP, 케이스에 합성이 들어갑니다.

                # 주관적으로, 'Y2K문제'와 같이 SL이 붙어 있는 경우도 명사로 구분하겠다고 결정했습니다. 따라서 SL + NNG도 합성합니다.
                
                # 숫자만 붙어 있는 경우는 '2026년'과 같이 미등록어가 아닐 확률이 높을 것이 간주, 합성하지 않습니다.
                # VA/VV + (VX 등) + ETM + NNG인 경우는 미등록어가 아닐 확률이 높을 것이라 간주, 합성하지 않습니다.
                
                if len(eojeol) == 1:
                    continue

                temp_eojeol = []

                for i in range(len(eojeol)):
                    
                    morpheme = eojeol[i]

                    # 어절의 첫 형태소인 경우 임시 어절에 바로 복사해 넣습니다.
                    if len(temp_eojeol) == 0:
                        temp_eojeol.append({'form': morpheme['form'], 'label': morpheme['label']})
                        continue
                    
                    # 임시 어절의 맨 뒤 형태소가 합성 가능한 종류이고, 현재 판정 중인 어절이 NNG or NNP면 합성합니다.
                    if temp_eojeol[-1]['label'] in ['XPN', 'NNG', 'NNP', 'MM', 'SL'] and morpheme['label'] in ['NNG', 'NNP']:
                        temp_eojeol[-1] = {'form': temp_eojeol[-1]['form'] + morpheme['form'], 'label': 'NNG'}
                        continue

                    # 합성 불가능한 경우 그냥 복사해 넣습니다.
                    temp_eojeol.append({'form': morpheme['form'], 'label': morpheme['label']})

                eojeol = temp_eojeol
                
                # 명사 N과 접미사열 S를 판별합니다.
                
                # 우선 첫 형태소 뒷부분의 'S가 될 수 있는 후보'를 떼어냅니다. 이때 SF, SP, SS, SO, SW는 포함하지 않습니다.
                if len(eojeol) == 1:
                    continue
                
                S_candidate = []
                for i in range(len(eojeol) - 1):
                    if eojeol[1+i]['label'] in ['SF', 'SP', 'SS', 'SO', 'SW']:
                        continue
                    S_candidate.append(eojeol[1+i])

                if not S_candidate:
                    continue

                # 만약 S 후보에 NNG, NNP, VV, VA같은 실질 형태소가 들어 있다면 버립니다. ('가정.학교.사회가'같은 엣지 케이스를 배제하기 위함입니다.)
                # 동시에 최종 S를 만들어 냅니다.
                temp_labels_list = []
                S = ''

                for S_morpheme in S_candidate:
                    temp_labels_list.append(S_morpheme['label'])
                    S += S_morpheme['form'] + '/' + S_morpheme['label'] + ' '

                if set(temp_labels_list) & {'NNG', 'NNP', 'VV', 'VA'}:
                    continue
                
                S = S.strip()
                S_total_count[S] += 1

                # 어절 첫 형태소가 N이 될 수 있는지 판별합니다.
                if eojeol[0]['label'] not in ['NNG', 'NNP']:
                    continue
                
                N = eojeol[0]['form']

                S_noun_count[S] += 1
                s_noun_map[S].add(N)
                noun_s_map[N].add(S)

                
# 공기 빈도를 계산한 뒤 최대 상위 K개의 SB를 저장합니다.
all_S = list(S_total_count.keys())


"""
# 작동은 하겠지만, 삼중 for문 안에 set & 연산 들어가 있는 게 시간복잡도가 정말 끔찍할 것 같은 느낌이 들어서 폐기한 알고리즘입니다.

for S in all_S:
    checked_SBs = set()
    SB_count_tuples = []

    for N in s_noun_map[S]:
        for SB in noun_s_map[N]:
            if SB in checked_SBs:
                continue

            ★-------------------------------★
            count = len(s_noun_map[S] & s_noun_map[SB])
            ★-------------------------------★

            SB_count_tuples.append((SB, count))
            checked_SBs.add(SB)
    
    sorted_tuples = sorted(SB_count_tuples, key=lambda p: p[1], reverse=True)
    sorted_tuples = sorted_tuples[:K]
    co_occurrence_frequencies[S] = copy.deepcopy(sorted_tuples)

"""

for i, S in enumerate(all_S):
    sb_counter = Counter()

    for N in s_noun_map[S]:
        for SB in noun_s_map[N]:
            if S == SB:
                continue

            sb_counter[SB] += 1

    top_k_SBs = sb_counter.most_common(K)
    co_occurrence_frequencies[S] = top_k_SBs

# print(co_occurrence_frequencies['를/JKO']) # 테스트용

print("JSON 파일로 저장 중...")

output_data = {
    "S_total_count": S_total_count,
    "S_noun_count": S_noun_count,
    "co_occurrence_frequencies": co_occurrence_frequencies, 
    "metadata": {
        "description": "미등록어 인식 모델 학습 데이터 (추론용)",
        "K_value": K
    }
}

with open(output_path, 'w', encoding='utf8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"저장 완료! ({output_path})")