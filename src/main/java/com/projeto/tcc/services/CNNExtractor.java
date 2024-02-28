package com.projeto.tcc.services;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

@Service
public class CNNExtractor {
    private final ComputationGraph vgg16;

    public CNNExtractor() throws IOException {
        // Carrega o modelo VGG16 pré-treinado uma única vez
        ZooModel zooModel = VGG16.builder().build();
        vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
    }
    public INDArray cnnFeaturesExtractor(String imagePath) throws IOException {
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
}
