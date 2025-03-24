class RuleBasedTransliterator:
    def __init__(self):
        self.initialize_variables()

    def initialize_variables(self):
        self.nVowels = 26
        self.consonants = []
        self.consonantsUni = []
        self.vowels = []
        self.vowelsUni = []
        self.vowelModifiersUni = []
        self.specialConsonants = []
        self.specialConsonantsUni = []
        self.specialCharUni = []
        self.specialChar = []

        self.vowelsUni.append("ඌ")
        self.vowels.append("oo")
        self.vowelModifiersUni.append("ූ")
        self.vowelsUni.append("ඕ")
        self.vowels.append("o\\)")
        self.vowelModifiersUni.append("ෝ")
        self.vowelsUni.append("ඕ")
        self.vowels.append("oe")
        self.vowelModifiersUni.append("ෝ")
        self.vowelsUni.append("ආ")
        self.vowels.append("aa")
        self.vowelModifiersUni.append("ා")
        self.vowelsUni.append("ආ")
        self.vowels.append("a\\)")
        self.vowelModifiersUni.append("ා")
        self.vowelsUni.append("ඈ")
        self.vowels.append("Aa")
        self.vowelModifiersUni.append("ෑ")
        self.vowelsUni.append("ඈ")
        self.vowels.append("A\\)")
        self.vowelModifiersUni.append("ෑ")
        self.vowelsUni.append("ඈ")
        self.vowels.append("ae")
        self.vowelModifiersUni.append("ෑ")
        self.vowelsUni.append("ඊ")
        self.vowels.append("ii")
        self.vowelModifiersUni.append("ී")
        self.vowelsUni.append("ඊ")
        self.vowels.append("i\\)")
        self.vowelModifiersUni.append("ී")
        self.vowelsUni.append("ඊ")
        self.vowels.append("ie")
        self.vowelModifiersUni.append("ී")
        self.vowelsUni.append("ඊ")
        self.vowels.append("ee")
        self.vowelModifiersUni.append("ී")
        self.vowelsUni.append("ඒ")
        self.vowels.append("ea")
        self.vowelModifiersUni.append("ේ")
        self.vowelsUni.append("ඒ")
        self.vowels.append("e\\)")
        self.vowelModifiersUni.append("ේ")
        self.vowelsUni.append("ඒ")
        self.vowels.append("ei")
        self.vowelModifiersUni.append("ේ")
        self.vowelsUni.append("ඌ")
        self.vowels.append("uu")
        self.vowelModifiersUni.append("ූ")
        self.vowelsUni.append("ඌ")
        self.vowels.append("u\\)")
        self.vowelModifiersUni.append("ූ")

        self.vowelsUni.append("ඖ")
        self.vowels.append("au")
        self.vowelModifiersUni.append("ෞ")

        self.vowelsUni.append("ඇ")
        self.vowels.append("\\a")
        self.vowelModifiersUni.append("ැ")

        self.vowelsUni.append("අ")
        self.vowels.append("a")
        self.vowelModifiersUni.append("")

        self.vowelsUni.append("ඇ")
        self.vowels.append("A")
        self.vowelModifiersUni.append("ැ")
        self.vowelsUni.append("ඉ")
        self.vowels.append("i")
        self.vowelModifiersUni.append("ි")
        self.vowelsUni.append("එ")
        self.vowels.append("e")
        self.vowelModifiersUni.append("ෙ")
        self.vowelsUni.append("උ")
        self.vowels.append("u")
        self.vowelModifiersUni.append("ු")
        self.vowelsUni.append("ඔ")
        self.vowels.append("o")
        self.vowelModifiersUni.append("ො")
        self.vowelsUni.append("ඓ")
        self.vowels.append("I")
        self.vowelModifiersUni.append("ෛ")

        self.specialConsonantsUni.append("ං")
        self.specialConsonants.append("\\n")

        self.specialConsonantsUni.append("ඃ")
        self.specialConsonants.append("\\h")
        self.specialConsonantsUni.append("ඞ")
        self.specialConsonants.append("\\N")
        self.specialConsonantsUni.append("ඍ")
        self.specialConsonants.append("\\R")
        # special characher Repaya
        self.specialConsonantsUni.append("ර්" + "\u200D")
        self.specialConsonants.append("R")
        self.specialConsonantsUni.append("ර්" + "\u200D")
        self.specialConsonants.append("\\r")

        self.consonantsUni.append("ඬ")
        self.consonants.append("nnd")

        self.consonantsUni.append("ඳ")
        self.consonants.append("nndh")

        self.consonantsUni.append("ඟ")
        self.consonants.append("nng")

        self.consonantsUni.append("ත")
        self.consonants.append("th")

        self.consonantsUni.append("ධ")
        self.consonants.append("dh")
        self.consonantsUni.append("ඝ")
        self.consonants.append("gh")
        self.consonantsUni.append("ච")
        self.consonants.append("ch")
        self.consonantsUni.append("ඵ")
        self.consonants.append("ph")
        self.consonantsUni.append("භ")
        self.consonants.append("bh")
        self.consonantsUni.append("ඣ")
        self.consonants.append("jh")
        self.consonantsUni.append("ෂ")
        self.consonants.append("sh")
        self.consonantsUni.append("ඥ")
        self.consonants.append("GN")
        self.consonantsUni.append("ඤ")
        self.consonants.append("KN")
        self.consonantsUni.append("ළු")
        self.consonants.append("Lu")
        self.consonantsUni.append("ඛ")
        self.consonants.append("kh")
        self.consonantsUni.append("ඨ")
        self.consonants.append("Th")
        self.consonantsUni.append("ඪ")
        self.consonants.append("Dh")
        self.consonantsUni.append("ශ")
        self.consonants.append("S")
        self.consonantsUni.append("ද")
        self.consonants.append("d")
        self.consonantsUni.append("ච")
        self.consonants.append("c")
        self.consonantsUni.append("ත")
        self.consonants.append("th")
        self.consonantsUni.append("ට")
        self.consonants.append("t")
        self.consonantsUni.append("ක")
        self.consonants.append("k")
        self.consonantsUni.append("ඩ")
        self.consonants.append("D")
        self.consonantsUni.append("න")
        self.consonants.append("n")
        self.consonantsUni.append("ප")
        self.consonants.append("p")
        self.consonantsUni.append("බ")
        self.consonants.append("b")
        self.consonantsUni.append("ම")
        self.consonants.append("m")
        self.consonantsUni.append("‍ය")
        self.consonants.append("\\u005C" + "y")
        self.consonantsUni.append("‍ය")
        self.consonants.append("Y")
        self.consonantsUni.append("ය")
        self.consonants.append("y")
        self.consonantsUni.append("ජ")
        self.consonants.append("j")
        self.consonantsUni.append("ල")
        self.consonants.append("l")
        self.consonantsUni.append("ව")
        self.consonants.append("v")
        self.consonantsUni.append("ව")
        self.consonants.append("w")
        self.consonantsUni.append("ස")
        self.consonants.append("s")
        self.consonantsUni.append("හ")
        self.consonants.append("h")
        self.consonantsUni.append("ණ")
        self.consonants.append("N")
        self.consonantsUni.append("ළ")
        self.consonants.append("L")
        self.consonantsUni.append("ඛ")
        self.consonants.append("K")
        self.consonantsUni.append("ඝ")
        self.consonants.append("G")
        self.consonantsUni.append("ඵ")
        self.consonants.append("P")
        self.consonantsUni.append("ඹ")
        self.consonants.append("B")
        self.consonantsUni.append("ෆ")
        self.consonants.append("f")
        self.consonantsUni.append("ග")
        self.consonants.append("g")
        # last because we need to ommit this in dealing with Rakaransha
        self.consonantsUni.append("ර")
        self.consonants.append("r")
        self.specialCharUni.append("ෲ")
        self.specialChar.append("ruu")
        self.specialCharUni.append("ෘ")
        self.specialChar.append("ru")
        # specialCharUni[2]="්‍ර" specialChar[2]="ra"

    def convert_text(self, text):
        s = ""
        r = ""
        v = ""
        # text = document.txtBox.box1.value;
        # special consonents
        for i in range(len(self.specialConsonants)):
            text = text.replace(self.specialConsonants[i], self.specialConsonantsUni[i])
        # consonents + special Chars
        for i in range(len(self.specialCharUni)):
            for j in range(len(self.consonants)):
                s = self.consonants[j] + self.specialChar[i]
                v = self.consonantsUni[j] + self.specialCharUni[i]
                # r = new RegExp(s, "g")
                r = s.replace(s + "/G", "")
                text = text.replace(r, v)

        # consonants + Rakaransha + vowel modifiers
        for j in range(len(self.consonants)):
            for i in range(len(self.vowels)):
                s = self.consonants[j] + "r" + self.vowels[i]
                v = self.consonantsUni[j] + "්‍ර" + self.vowelModifiersUni[i]
                # r = new RegExp(s, "g")
                r = s.replace(s + "/G", "")
                text = text.replace(r, v)
            s = self.consonants[j] + "r"
            v = self.consonantsUni[j] + "්‍ර"
            # r = new RegExp(s, "g")
            r = s.replace(s + "/G", "")
            text = text.replace(r, v)
        # consonents + vowel modifiers
        for i in range(len(self.consonants)):
            for j in range(self.nVowels):
                s = self.consonants[i] + self.vowels[j]
                v = self.consonantsUni[i] + self.vowelModifiersUni[j]
                # r = new RegExp(s, "g")
                r = s.replace(s + "/G", "")
                text = text.replace(r, v)

        # consonents + HAL
        for i in range(len(self.consonants)):
            r = self.consonants[i].replace(self.consonants[i] + "/G", "")
            text = text.replace(r, self.consonantsUni[i] + "්")
        # vowels
        for i in range(len(self.vowels)):
            # r = new RegExp(vowels[i], "g")
            r = self.vowels[i].replace(self.vowels[i] + "/G", "")
            text = text.replace(r, self.vowelsUni[i])
        return text
