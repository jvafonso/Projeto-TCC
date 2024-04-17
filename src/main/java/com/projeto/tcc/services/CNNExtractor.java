package com.projeto.tcc.services;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.*;

@Service
@Slf4j
public class CNNExtractor {
    private final ComputationGraph vgg16;

    public CNNExtractor() throws IOException {
        // Carrega o modelo VGG16 pré-treinado uma única vez
        ZooModel zooModel = VGG16.builder().build();
        vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        vgg16.initGradientsView();
    }

    public List<INDArray> cnnFeaturesExtractorBatch(List<String> imagePaths) throws IOException {
        log.info("Extração CNN em lote");
        List<INDArray> featuresList = new ArrayList<>();

        for (String imagePath : imagePaths) {
            INDArray frame = loadImageAndPreProcess(imagePath);

            // Processa a imagem individualmente
            INDArray singleFeature = vgg16.feedForward(frame, false).get("fc2");

            // Evita vazamentos de memória e problemas de espaço de trabalho INDArrays
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

            featuresList.add(singleFeature);
        }

        return featuresList;
    }



    // Função para carregar e pré-processar uma imagem
    public INDArray loadImageAndPreProcess(String imagePath) throws IOException {
        /*
        Redimensionamento da Imagem: A imagem é redimensionada para o tamanho esperado pela rede VGG16, que é 224x224 pixels. Isso é feito pela classe NativeImageLoader ao carregar a imagem.
        Subtração da Média: A média de cada canal de cor (RGB) é subtraída de cada pixel. Esse valor médio é específico para o conjunto de dados ImageNet, no qual o modelo VGG16 foi pré-treinado. Essa etapa é importante para centralizar os dados em torno de zero, o que ajuda na convergência do treinamento da rede.
        Reordenamento dos Canais: Os canais de cor são reordenados. Por padrão, muitas bibliotecas de processamento de imagens, incluindo o OpenCV, carregam imagens no formato BGR (Azul, Verde, Vermelho). No entanto, a rede VGG16 espera que as imagens estejam no formato RGB. Portanto, a ordem dos canais é alterada para corresponder à expectativa da rede.
         */
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(new File(imagePath));
        VGG16ImagePreProcessor preProcessor = new VGG16ImagePreProcessor();
        preProcessor.transform(image);
        return image;
    }

    public static double compareFeatures(INDArray features1, INDArray features2) {
        // Normalizar os vetores de características
        INDArray normFeatures1 = features1.div(features1.norm2Number());
        INDArray normFeatures2 = features2.div(features2.norm2Number());
        int alpha = 5;

        // Calcular a distância euclidiana entre os vetores normalizados
        INDArray diff = normFeatures1.sub(normFeatures2);
        double distanciaEuclidiana = diff.norm2Number().doubleValue();

        // Normalização para o intervalo [0.1, 1] usando uma função sigmoide ajustada
        double similaridade = 0.9 / (1 + Math.exp(alpha * distanciaEuclidiana)) + 0.1;

        return similaridade;
    }

    public static INDArray stringToINDArray(String str) {
        // Convertendo a string de volta para um INDArray
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new StringReader(str))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                rows.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        double[][] data = rows.toArray(new double[0][]);
        return Nd4j.create(data);
    }

    public static INDArray calculateAverageIndArray(List<INDArray> arrays) {
        if (arrays == null || arrays.isEmpty()) {
            throw new IllegalArgumentException("A lista não pode ser vazia.");
        }
        // Inicializa um INDArray para armazenar a soma
        INDArray sum = Nd4j.zeros(arrays.get(0).shape());
        // Soma todos os elementos da lista
        for (INDArray array : arrays) {
            sum.addi(array);
        }
        // Calcula a média
        return sum.divi(arrays.size());
    }

    public List<INDArray> groupFramesCNN(File descriptorFile, double similarityThreshold, double samplingPercentage) throws IOException {
        List<CNNGroup> groups = new ArrayList<>();
        List<INDArray> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            String line;
            StringBuilder descriptorString = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    if (descriptorString.length() > 0) {
                        INDArray descriptor = stringToINDArray(descriptorString.toString());
                        addToCNNGroup(groups, descriptor, similarityThreshold);
                        descriptorString.setLength(0); // Clear the StringBuilder for the next descriptor
                    }
                } else {
                    descriptorString.append(line).append("\n");
                }
            }
        }

        for (CNNGroup group : groups) {
            group.descriptors.sort(Comparator.comparingDouble(d -> compareFeatures(group.average, d)));
            Collections.reverse(group.descriptors);  // Para ter os mais similares primeiro
            int elementsToSample = (int) (group.descriptors.size() * (samplingPercentage / 100.0));
            if (elementsToSample == 0 && !group.descriptors.isEmpty()) {
                elementsToSample = 1;
            }
            selectedDescriptors.addAll(group.descriptors.subList(0, elementsToSample));
            log.info(elementsToSample + " quadros adicionados à amostra do grupo.");
        }

        return selectedDescriptors;
    }

    private void addToCNNGroup(List<CNNGroup> groups, INDArray descriptor, double similarityThreshold) {
        for (CNNGroup group : groups) {
            if (group.isSimilar(descriptor, similarityThreshold)) {
                group.add(descriptor);
                log.info("Quadro adicionado ao grupo");
                return;
            }
        }
        log.info("Novo grupo criado");
        groups.add(new CNNGroup(descriptor));
    }

    private static class CNNGroup {
        private List<INDArray> descriptors = new ArrayList<>();
        private INDArray average;

        CNNGroup(INDArray descriptor) {
            add(descriptor);
        }

        void add(INDArray descriptor) {
            descriptors.add(descriptor);
            updateAverage(descriptor);
        }

        boolean isSimilar(INDArray descriptor, double similarityThreshold) {
            if (average == null) {
                return true;
            }
            double similarity = compareFeatures(average, descriptor);
            log.info("Distancia: {}, similarityThreshold: {}", similarity, similarityThreshold);
            return similarity < similarityThreshold;
        }

        private void updateAverage(INDArray newDescriptor) {
            if (average == null) {
                average = newDescriptor.dup();
            } else {
                average = average.add(newDescriptor).div(descriptors.size());
            }
        }
    }

    public List<INDArray> sampleRandomCNNDescriptors(File descriptorFile, double samplingPercentage) throws IOException {
        List<INDArray> allDescriptors = new ArrayList<>();

        // Lendo e parseando todos os descritores do arquivo
        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            String line;
            StringBuilder descriptorString = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    if (descriptorString.length() > 0) {
                        INDArray descriptor = stringToINDArray(descriptorString.toString());
                        allDescriptors.add(descriptor);
                        descriptorString.setLength(0);  // Limpa o StringBuilder para o próximo descritor
                    }
                } else {
                    descriptorString.append(line).append("\n");
                }
            }
        }

        // Calculando a quantidade de descritores a serem amostrados
        int totalDescriptors = allDescriptors.size();
        int sampleSize = (int) (totalDescriptors * samplingPercentage / 100.0);
        if (sampleSize == 0 && !allDescriptors.isEmpty()) {
            sampleSize = 1;  // Garante que pelo menos um elemento seja selecionado
        }

        // Selecionando aleatoriamente os descritores
        List<INDArray> sampledDescriptors = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < sampleSize; i++) {
            int randomIndex = random.nextInt(allDescriptors.size());
            sampledDescriptors.add(allDescriptors.get(randomIndex));
            allDescriptors.remove(randomIndex);  // Para evitar amostragem repetida
        }

        return sampledDescriptors;
    }


    public List<INDArray> sampleFramesBySecondCNN(File descriptorFile, double samplingPercentage) throws IOException {
        List<INDArray> allDescriptors = new ArrayList<>();

        // Lendo e parseando todos os descritores do arquivo
        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            String line;
            StringBuilder descriptorString = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    if (descriptorString.length() > 0) {
                        INDArray descriptor = stringToINDArray(descriptorString.toString());
                        allDescriptors.add(descriptor);
                        descriptorString.setLength(0);  // Limpa o StringBuilder para o próximo descritor
                    }
                } else {
                    descriptorString.append(line).append("\n");
                }
            }
        }

        List<INDArray> sampledDescriptors = new ArrayList<>();
        int framesPerSecond = 30; // Quantidade de quadros por segundo
        int sampleSize = (int) (framesPerSecond * (samplingPercentage / 100.0));

        for (int i = 0; i < allDescriptors.size(); i += framesPerSecond) {
            int end = Math.min(i + sampleSize, allDescriptors.size());
            for (int j = i; j < end; j++) {
                sampledDescriptors.add(allDescriptors.get(j));
            }
        }

        return sampledDescriptors;
    }





}
