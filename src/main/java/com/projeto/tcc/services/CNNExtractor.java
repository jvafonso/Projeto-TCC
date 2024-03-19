package com.projeto.tcc.services;
import lombok.extern.slf4j.Slf4j;
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
    }
    public INDArray cnnFeaturesExtractor(String imagePath) throws IOException {
        log.info("Extração CNN");
        // Carrega e faz o pre-processamento da imagem para extração das caracteristicas pela CNN
        INDArray frame = loadImageAndPreProcess(imagePath);
        // Extrai características das imagens usando a camada 'fc2' do VGG16
        INDArray features = vgg16.feedForward(frame, false).get("fc2");
        return features;
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

    // Função para comparar características extraídas de duas imagens
    public double compareFeatures(INDArray features1, INDArray features2) {
        log.info("Comparação CNN");
        // Calcula a distância euclidiana entre os dois vetores de características
        // Calcula a diferença elemento a elemento entre os dois vetores de características
        INDArray diff = features1.sub(features2);
        // Calcula o quadrado de cada elemento do vetor de diferença
        INDArray squaredDiff = diff.mul(diff);
        // Soma todos os elementos do vetor de quadrados para obter o quadrado da distância euclidiana
        double squaredDistance = squaredDiff.sumNumber().doubleValue();
        // Calcula a raiz quadrada para obter a distância euclidiana
        return Math.sqrt(squaredDistance);
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

    public List<INDArray> groupFramesCNN(File descriptorFile, double similarityThreshold) throws IOException {
        List<List<INDArray>> groups = new ArrayList<>();
        List<INDArray> selectedDescriptors = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(descriptorFile))) {
            String line;
            StringBuilder descriptorString = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    if (descriptorString.length() > 0) {
                        INDArray descriptor = stringToINDArray(descriptorString.toString());
                        boolean addedToGroup = false;
                        for (List<INDArray> group : groups) {
                            INDArray groupDescriptor = calculateAverageIndArray(group);
                            double similarity = compareFeatures(groupDescriptor, descriptor);
                            if (similarity > similarityThreshold) {
                                group.add(descriptor);
                                addedToGroup = true;
                                break;
                            }
                        }
                        if (!addedToGroup) {
                            List<INDArray> newGroup = new ArrayList<>();
                            newGroup.add(descriptor);
                            groups.add(newGroup);
                        }
                        descriptorString.setLength(0); // Clear the StringBuilder for the next descriptor
                    }
                } else {
                    descriptorString.append(line).append("\n");
                }
            }
        }

        for (List<INDArray> group : groups) {
            INDArray groupDescriptor = calculateAverageIndArray(group);
            INDArray selectedDescriptor = null;
            double maxSimilarity = -1;
            for (INDArray descriptor : group) {
                double similarity = compareFeatures(groupDescriptor, descriptor);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    selectedDescriptor = descriptor;
                }
            }
            if (selectedDescriptor != null) {
                selectedDescriptors.add(selectedDescriptor);
            }
        }
        return selectedDescriptors;
    }





}
