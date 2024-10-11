# TRIBE (Michael Chen, Justin Liaw, John Jenness, Chris Kim)

---

### Overview

#### Project Title
**A Quest Called TRIBE: Clustering Malware Families for Enhanced Triage and Analysis**

#### Project Objective
The project seeks to improve machine learning-based malware classification through a novel model called the Tribal Relation Inferential Binary Encoder (TRIBE). This model clusters malware into ‘tribes’ based on binary data, facilitating faster and more accurate malware classification in cybersecurity responses.

#### Problem Statement
Traditional antivirus (AV) methods often struggle to keep up with evolving malware. By clustering malware into tribes, TRIBE aims to enhance the speed and accuracy of malware classification, addressing a critical need in incident response workflows within security operations centers (SOCs).

#### Importance
The project responds to the growing challenge of malware, especially within critical sectors such as government, healthcare, and the military. The TRIBE model could significantly reduce response times to malware threats, a critical factor in national security contexts.

---

### Market Research/Lit Review

#### Existing Processes
Currently, AV methods rely on signature and heuristic-based detection, which are limited by the rapid evolution of malware. These traditional approaches struggle to generalize across the vast diversity of malware.

#### Market Research
Research into machine learning for malware classification is active, but current solutions do not adequately address AV family labeling's complexity. TRIBE’s tribal clustering proposes a more efficient and adaptable approach compared to traditional AV family labels.

---

### Proposed Design and Architecture

#### User Types/Personas
1. **Cyber Analysts** - Primary users who will leverage the model for rapid malware classification.
2. **System Administrators** - Secondary users who will benefit from reduced system vulnerabilities.

#### System Design
The TRIBE model utilizes a transformer-based sequence-to-sequence autoencoder to compress and reconstruct malware binaries, generating representations for clustering into tribes.

#### System Architecture
The system consists of:
1. **Autoencoder** - Encodes malware binaries into latent-space representations.
2. **Clustering Algorithms** - Groups malware into tribes using K-means, K-modes, and DBSCAN algorithms.

---

### Project Management

#### Preliminary Release Plan
The project will undergo multiple development and testing phases:
- **Phase 1:** Data Collection and Preprocessing.
- **Phase 2:** Development of Transformer-based Autoencoder.
- **Phase 3:** Clustering and Testing of Malware Families.
- **Phase 4:** Model Validation on AV Labeling Systems.

#### Product Backlog
1. **Data Collection and Preprocessing**
   - Collection from MalwareBazaar
   - ClamAV family labeling
2. **Model Development**
   - Autoencoder architecture search and optimization
3. **Clustering Analysis**
   - Apply and test multiple clustering algorithms
4. **Model Validation**
   - Evaluate TRIBE against traditional AV labeling

#### Risks and Mitigations
1. **Computational Limitations** - Mitigated by using a downsized, balanced dataset.
2. **Class Imbalance** - Addressed through random sampling techniques to balance malware family distributions.
3. **Dependency on Autoencoder Quality** - The success of TRIBE relies on accurate binary encoding. A robust architecture search will ensure optimal performance.

#### Faculty SMEs
- **Primary Advisor:** Dr. Dane Brown, Cyber Science Department
- **Secondary Advisor:** CDR Edgar Jatho, Computer Science Department

---

### Admin/Fine Print

#### Resources Required
1. **GFE/GFI**: Access to MalwareBazaar and ClamAV.
2. **Additional Resources**: High-performance computing with GPUs that are already secured and set up off USNA network to be used for Malware dataset.

#### Customer Meeting Requirements/Plan
Weekly meetings with project advisors for status updates and guidance.

#### Acceptance Window
Scheduled completion by the end of the academic year.

#### Code Delivery
The code will be delivered via GitLab, with access provided to project advisors.

#### Usage License
The project follows the standard usage license outlined in Paragraph 10 of the USNA CS Capstone Instruction.

#### Termination Clause
In the event of project termination, a "recovery plan" will be implemented to enhance and already existing model.

---

