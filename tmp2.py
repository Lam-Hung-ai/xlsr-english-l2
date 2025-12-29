from g2p_en import G2p

from convertor import arpabet2ipa

text  = "this is a test"
argabet = G2p()(text)
print("ARPABET:", ' '.join(argabet))
ipa = arpabet2ipa(' '.join(argabet))
print("IPA:", ipa)