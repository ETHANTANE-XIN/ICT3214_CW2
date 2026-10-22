"""IEEE-style references (each verified against publisher or indexing records) and a citation helper.

Write citations in the text as [@key] or [@key1,@key2]; ``Citer.cite`` replaces them with IEEE
numbers in order of first appearance and ``Citer.reference_list`` returns the matching list.
"""
import re

REFS = {
    "hady": 'A. A. Hady, A. Ghubaish, T. Salman, D. Unal, and R. Jain, "Intrusion detection system for healthcare systems using medical and network data: A comparison study," *IEEE Access*, vol. 8, pp. 106576–106584, 2020, doi: 10.1109/ACCESS.2020.3000421.',
    "ghubaish": 'A. Ghubaish et al., "Recent advances in the Internet-of-Medical-Things (IoMT) systems security," *IEEE Internet Things J.*, vol. 8, no. 11, pp. 8707–8718, Jun. 2021.',
    "halperin": 'D. Halperin et al., "Pacemakers and implantable cardiac defibrillators: Software radio attacks and zero-power defenses," in *Proc. IEEE Symp. Security and Privacy*, Oakland, CA, USA, 2008, pp. 129–142.',
    "meidan": 'Y. Meidan et al., "N-BaIoT: Network-based detection of IoT botnet attacks using deep autoencoders," *IEEE Pervasive Comput.*, vol. 17, no. 3, pp. 12–22, 2018.',
    "mirsky": 'Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, "Kitsune: An ensemble of autoencoders for online network intrusion detection," in *Proc. Netw. Distrib. Syst. Security Symp. (NDSS)*, San Diego, CA, USA, 2018. [Online]. Available: https://arxiv.org/abs/1802.09089',
    "vaccari": 'I. Vaccari, G. Chiola, M. Aiello, M. Mongelli, and E. Cambiaso, "MQTTset, a new dataset for machine learning techniques on MQTT," *Sensors*, vol. 20, no. 22, Art. no. 6578, 2020, doi: 10.3390/s20226578.',
    "hindy": 'H. Hindy, E. Bayne, M. Bures, R. Atkinson, C. Tachtatzis, and X. Bellekens, "Machine learning based IoT intrusion detection system: An MQTT case study (MQTT-IoT-IDS2020 dataset)," in *Selected Papers From the 12th Int. Netw. Conf. (INC 2020)*, B. Ghita and S. Shiaeles, Eds., Lecture Notes in Networks and Systems, vol. 180. Cham, Switzerland: Springer, 2021, pp. 73–84, doi: 10.1007/978-3-030-64758-2_6.',
    "ahmed": 'M. Ahmed, S. Byreddy, A. Nutakki, L. F. Sikos, and P. Haskell-Dowland, "ECU-IoHT: A dataset for analyzing cyberattacks in Internet of Health Things," *Ad Hoc Netw.*, vol. 122, Art. no. 102621, 2021, doi: 10.1016/j.adhoc.2021.102621.',
    "sharafaldin": 'I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. 4th Int. Conf. Inf. Syst. Security Privacy (ICISSP)*, Funchal, Portugal, 2018, pp. 108–116, doi: 10.5220/0006639801080116.',
    "moustafa": 'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," in *Proc. Mil. Commun. Inf. Syst. Conf. (MilCIS)*, Canberra, ACT, Australia, 2015, pp. 1–6, doi: 10.1109/MilCIS.2015.7348942.',
    "sommer": 'R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in *Proc. IEEE Symp. Security and Privacy*, Oakland, CA, USA, 2010, pp. 305–316, doi: 10.1109/SP.2010.25.',
    "arp": 'D. Arp et al., "Dos and don\'ts of machine learning in computer security," in *Proc. 31st USENIX Security Symp.*, Boston, MA, USA, 2022, pp. 3971–3988.',
    "tao": 'F. Tao, H. Zhang, A. Liu, and A. Y. C. Nee, "Digital twin in industry: State-of-the-art," *IEEE Trans. Ind. Informat.*, vol. 15, no. 4, pp. 2405–2415, Apr. 2019, doi: 10.1109/TII.2018.2873186.',
    "breiman": 'L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.',
    "chen": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794.',
    "liu": 'F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.',
    "lundberg": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA, USA, 2017, pp. 4765–4774.',
    "pedregosa": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
    "chandola": 'V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Comput. Surv.*, vol. 41, no. 3, Art. no. 15, 2009.',
    "wustl_ds": '"WUSTL EHMS 2020 dataset for Internet of Medical Things (IoMT) cybersecurity research," Washington Univ. in St. Louis, 2020. [Online]. Available: https://www.cse.wustl.edu/~jain/ehms/index.html',
}

CITE = re.compile(r"\[@([a-z_0-9,@ ]+)\]")


class Citer:
    def __init__(self):
        self.order: list[str] = []

    def _num(self, key):
        key = key.strip().lstrip("@")
        if key not in REFS:
            raise KeyError(f"unknown reference {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def cite(self, text: str) -> str:
        def rep(m):
            nums = sorted(self._num(k) for k in m.group(1).split(","))
            return "[" + "], [".join(str(n) for n in nums) + "]"
        return CITE.sub(rep, text)

    def reference_list(self):
        return [REFS[k] for k in self.order]
