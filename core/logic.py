# 1. Bảng mã ARPABET sang IPA
ARPABET2IPA = {
    'AA':'ɑ','AE':'æ','AH':'ʌ','AO':'ɔ','IX':'ɨ','AW':'aʊ','AX':'ə','AXR':'ɚ','AY':'aɪ',
    'EH':'ɛ','ER':'ɝ','EY':'eɪ','IH':'ɪ','IY':'i','OW':'oʊ','OY':'ɔɪ','UH':'ʊ','UW':'u',
    'UX':'ʉ','B':'b','CH':'tʃ','D':'d','DH':'ð','EL':'l̩','EM':'m̩','EN':'n̩','F':'f',
    'G':'ɡ','HH':'h','H':'h','JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ',
    'NX':'ɾ̃','P':'p','Q':'ʔ','R':'ɹ','S':'s','SH':'ʃ','T':'t','TH':'θ','V':'v',
    'W':'w','WH':'ʍ','Y':'j','Z':'z','ZH':'ʒ','DX':'ɾ'
}

# 2. Bảng mã trọng âm (Stress markers) cho tiếng Anh
ENGLISH_STRESS = { '0':'', '1':'ˈ', '2':'ˌ' }

def string2symbols(string, symbols):
    """
    Tách chuỗi ARPABET thành danh sách các ký hiệu hợp lệ.
    Ví dụ: "BAET" -> ["B", "AE", "T"]
    """
    N = len(string)
    oovcost = len(string)
    maxsym = max(len(k) for k in symbols)
    lattice = [(0, 0, "", True)]

    for n in range(1, N + 1):
        lattice.append((oovcost + lattice[n - 1][0], n - 1, string[(n - 1) : n], False))
        for m in range(1, min(n + 1, maxsym + 1)):
            if string[(n - m) : n] in symbols and 1 + lattice[n - m][0] < lattice[n][0]:
                lattice[n] = (1 + lattice[n - m][0], n - m, string[(n - m) : n], True)

    tl = []
    n = N
    while n > 0:
        tl.append(lattice[n][2])
        n = lattice[n][1]
    return tl[::-1]

def arpabet2ipa(arpabet_string, get_stress=False):
    """
    Chuyển đổi chuỗi ARPABET sang IPA.
    Hỗ trợ cả chuỗi có dấu cách (B AE1 T) hoặc dính liền (BAET).
    """
    # Tạo từ điển đầy đủ bao gồm cả trọng âm
    full_mapping = ARPABET2IPA.copy()
    full_mapping.update(ENGLISH_STRESS)

    # Tách các ký hiệu (tokens)
    if " " in arpabet_string:
        arpabet_symbols = arpabet_string.split()
    else:
        arpabet_symbols = string2symbols(arpabet_string.upper(), full_mapping.keys())

    # Chuyển đổi từng ký hiệu sang IPA
    res = ""
    for sym in arpabet_symbols:
        # Xử lý trường hợp ký hiệu có kèm số trọng âm (vd: AE1, AH0)
        main_sym = sym
        stress = ""
        if len(sym) > 1 and sym[-1].isdigit():
            main_sym = sym[:-1]
            stress = ENGLISH_STRESS.get(sym[-1], "")

        # Lấy giá trị IPA, nếu không có thì giữ nguyên ký hiệu gốc
        ipa_val = ARPABET2IPA.get(main_sym, main_sym)
        if get_stress:
            res += stress + ipa_val
        else:
            res += ipa_val

    return res

def align_phonemes(target_ipa, hypothesis_ipa):
    """
    So sánh chuỗi IPA chuẩn và chuỗi IPA người dùng đọc.
    Trả về danh sách các cặp (âm chuẩn, âm thực tế, trạng thái).
    """
    n, m = len(target_ipa), len(hypothesis_ipa)
    # Khởi tạo ma trận khoảng cách
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target_ipa[i-1] == hypothesis_ipa[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

    # Truy vết (Backtracking) để tìm các lỗi cụ thể
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and target_ipa[i-1] == hypothesis_ipa[j-1]:
            alignment.append((target_ipa[i-1], hypothesis_ipa[j-1], "Correct"))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((target_ipa[i-1], hypothesis_ipa[j-1], "Substitution"))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            alignment.append((target_ipa[i-1], "-", "Deletion"))
            i -= 1
        else:
            alignment.append(("-", hypothesis_ipa[j-1], "Insertion"))
            j -= 1

    return alignment[::-1]