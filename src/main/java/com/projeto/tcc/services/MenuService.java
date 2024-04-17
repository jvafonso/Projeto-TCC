package com.projeto.tcc.services;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ExecutionException;

@Service
@Slf4j
public class MenuService {

    @Autowired
    private FrameService2 frameService;

    @Autowired
    private FrameExtractor frameExtractor;

    @Autowired
    private PointsFileGenerator pointsFileGenerator;

    public void startMenu() throws IOException {
        Scanner scanner = new Scanner(System.in);
        double similarityThreshold = 0;
        double samplingPorcentage = 0;

        while (true) {
            System.out.println("\nEscolha uma opção:");
            System.out.println("1 - Definir taxa de similaridade e porcentagem de seleção de quadros para a amostra");
            System.out.println("2 - Separar frames de um vídeo");
            System.out.println("3 - Extrair caracteristicas de um grupo de frames");
            System.out.println("4 - Gerar points file de video completo");
            System.out.println("5 - Agrupar frames");
            System.out.println("6 - Gerar frames das amostras");
            System.out.println("7 - Amostra aleatoria");
            System.out.println("8 - Amostra por segundo");
            System.out.println("9 - Sair");
            System.out.print("Opção: ");
            int option = scanner.nextInt();
            scanner.nextLine(); // Consume newline left-over

            switch (option) {
                case 1:
                    System.out.print("Insira a taxa de similaridade (0-100): ");
                    similarityThreshold = scanner.nextDouble() / 100;
                    if (similarityThreshold < 0 || similarityThreshold > 1) {
                        System.out.println("A taxa de similaridade deve estar entre 0 e 100.");
                        break;
                    }
                    System.out.print("Insira a porcentagem de seleção de quadros para a amostra (0-100): ");
                    samplingPorcentage = scanner.nextDouble();
                    break;
                case 2:
                    System.out.print("Insira o caminho do arquivo de vídeo de entrada: ");
                    String videoPath = scanner.nextLine();
                    try {
                        frameExtractor.samplingFrames(videoPath);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
                case 3:
                    long startTimeExtraction = System.currentTimeMillis();
                    File framesDir = new File("D:\\UFU\\tcc_video_frames\\frames");
                    File[] subdirs = framesDir.listFiles(File::isDirectory);
                    if (subdirs != null) {
                        for (int i = 0; i < subdirs.length; i++) {
                            System.out.println((i + 1) + " - " + subdirs[i].getName());
                        }
                        System.out.print("Escolha o conjunto de frames para extrair características: ");
                        int dirChoice = scanner.nextInt();
                        if (dirChoice > 0 && dirChoice <= subdirs.length) {
                            String chosenFramesPath = subdirs[dirChoice - 1].getAbsolutePath();
                            try {
                                frameExtractor.extractFeaturesV2(chosenFramesPath ,subdirs[dirChoice - 1].getName());
                            } catch (IOException | InterruptedException | ExecutionException e) {
                                e.printStackTrace();
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhum conjunto de frames encontrado.");
                    }
                    long endTimeExtraction = System.currentTimeMillis();
                    log.info("Tempo de execução da extração de caracteristicas: " + (endTimeExtraction - startTimeExtraction) + " ms");
                    break;
                case 4:
                    System.out.println("Escolha uma subpasta de descritores:");
                    File descriptorsBaseDirPoints = new File("D:\\UFU\\tcc_video_frames\\descriptores");
                    File[] extractionSubdirsPoints = descriptorsBaseDirPoints.listFiles(File::isDirectory);

                    if (extractionSubdirsPoints != null && extractionSubdirsPoints.length > 0) {
                        for (int i = 0; i < extractionSubdirsPoints.length; i++) {
                            System.out.println((i + 1) + " - " + extractionSubdirsPoints[i].getName());
                        }
                        System.out.print("Escolha a subpasta de descritores para o agrupamento: ");
                        int dirChoice = scanner.nextInt();
                        scanner.nextLine();
                        if (dirChoice > 0 && dirChoice <= extractionSubdirsPoints.length) {
                            File chosenDescriptorSubdir = extractionSubdirsPoints[dirChoice - 1];
                            String[] possibleDescriptorFiles = {"descritoresHOG.txt", "descritoresORB.txt", "descritoresCNN.txt"};
                            boolean fileFound = false;
                            for (String descFileName : possibleDescriptorFiles) {
                                File descriptorFile = new File(chosenDescriptorSubdir, descFileName);
                                if (descriptorFile.exists()) {
                                    String chosenDescriptorsFilePath = descriptorFile.getAbsolutePath();
                                    try {
                                        File file = new File(chosenDescriptorsFilePath);
                                        String fileName = file.getName();
                                        String extractorName = "";
                                        if (fileName.contains("HOG")) {
                                            extractorName = "HOG";
                                        } else if (fileName.contains("ORB")) {
                                            extractorName = "ORB";
                                        } else if (fileName.contains("CNN")) {
                                            extractorName = "CNN";
                                        } else {
                                        throw new IllegalArgumentException("Tipo de descritor desconhecido.");
                                        }
                                        frameService.pointsFileHoleSet(file, "D:\\UFU\\tcc_video_frames\\cojuntosCompletos\\extracao" + extractorName + "\\pointsFile", extractorName);
                                        fileFound = true;
                                        break;
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            }
                            if (!fileFound) {
                                System.out.println("Nenhum arquivo de descritores válido foi encontrado na subpasta escolhida.");
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhuma subpasta de descritores encontrada.");
                    }
                    break;
                case 5:
                    long startTimeGroup = System.currentTimeMillis();
                    if (similarityThreshold == 0 || samplingPorcentage == 0) {
                        System.out.println("A taxa de similaridade ou a porcentagem de seleção da amostra não podem ser 0 escolha a opção 1 e determine um valor.");
                        break;
                    }
                    System.out.println("Escolha uma subpasta de descritores:");
                    File descriptorsBaseDir = new File("D:\\UFU\\tcc_video_frames\\descriptores");
                    File[] extractionSubdirs = descriptorsBaseDir.listFiles(File::isDirectory);

                    if (extractionSubdirs != null && extractionSubdirs.length > 0) {
                        for (int i = 0; i < extractionSubdirs.length; i++) {
                            System.out.println((i + 1) + " - " + extractionSubdirs[i].getName());
                        }
                        System.out.print("Escolha a subpasta de descritores para o agrupamento: ");
                        int dirChoice = scanner.nextInt();
                        scanner.nextLine();
                        if (dirChoice > 0 && dirChoice <= extractionSubdirs.length) {
                            File chosenDescriptorSubdir = extractionSubdirs[dirChoice - 1];
                            String[] possibleDescriptorFiles = {"descritoresHOG.txt", "descritoresORB.txt", "descritoresCNN.txt"};
                            boolean fileFound = false;
                            for (String descFileName : possibleDescriptorFiles) {
                                File descriptorFile = new File(chosenDescriptorSubdir, descFileName);
                                if (descriptorFile.exists()) {
                                    String chosenDescriptorsFilePath = descriptorFile.getAbsolutePath();
                                    try {
                                        frameService.groupFrames(chosenDescriptorsFilePath, similarityThreshold, samplingPorcentage);
                                        fileFound = true;
                                        break;
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            }
                            if (!fileFound) {
                                System.out.println("Nenhum arquivo de descritores válido foi encontrado na subpasta escolhida.");
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhuma subpasta de descritores encontrada.");
                    }
                    long endTimeGroup = System.currentTimeMillis();
                    log.info("Tempo de execução do agrupamento de quadros: " + (endTimeGroup - startTimeGroup) + " ms");
                    break;
                case 6:
                    System.out.print("Insira o caminho do arquivo points do vídeo completo: ");
                    String pointsComplete = scanner.nextLine();
                    System.out.print("Insira o caminho do arquivo points da amostra: ");
                    String samplePoints = scanner.nextLine();
                    System.out.print("Insira o caminho do diretorio com os quadros: ");
                    String framesPath = scanner.nextLine();
                    System.out.print("Insira o caminho do diretorio que vai armazenar o resultado: ");
                    String resultPath = scanner.nextLine();
                    try {
                        pointsFileGenerator.extractAndCopyFrames(pointsComplete, samplePoints, framesPath,resultPath);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
                case 7:
                    if (similarityThreshold == 0 || samplingPorcentage == 0) {
                        System.out.println("A taxa de similaridade ou a porcentagem de seleção da amostra não podem ser 0 escolha a opção 1 e determine um valor.");
                        break;
                    }
                    System.out.println("Escolha uma subpasta de descritores:");
                    File descriptorsBaseDir2 = new File("D:\\UFU\\tcc_video_frames\\descriptores");
                    File[] extractionSubdirs2 = descriptorsBaseDir2.listFiles(File::isDirectory);

                    if (extractionSubdirs2 != null && extractionSubdirs2.length > 0) {
                        for (int i = 0; i < extractionSubdirs2.length; i++) {
                            System.out.println((i + 1) + " - " + extractionSubdirs2[i].getName());
                        }
                        System.out.print("Escolha a subpasta de descritores para o agrupamento: ");
                        int dirChoice = scanner.nextInt();
                        scanner.nextLine();
                        if (dirChoice > 0 && dirChoice <= extractionSubdirs2.length) {
                            File chosenDescriptorSubdir = extractionSubdirs2[dirChoice - 1];
                            String[] possibleDescriptorFiles = {"descritoresHOG.txt", "descritoresORB.txt", "descritoresCNN.txt"};
                            boolean fileFound = false;
                            for (String descFileName : possibleDescriptorFiles) {
                                File descriptorFile = new File(chosenDescriptorSubdir, descFileName);
                                if (descriptorFile.exists()) {
                                    String chosenDescriptorsFilePath = descriptorFile.getAbsolutePath();
                                    try {
                                        frameService.randomSample(chosenDescriptorsFilePath, samplingPorcentage);
                                        fileFound = true;
                                        break;
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            }
                            if (!fileFound) {
                                System.out.println("Nenhum arquivo de descritores válido foi encontrado na subpasta escolhida.");
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhuma subpasta de descritores encontrada.");
                    }
                    break;
                case 8:
                    if (similarityThreshold == 0 || samplingPorcentage == 0) {
                        System.out.println("A taxa de similaridade ou a porcentagem de seleção da amostra não podem ser 0 escolha a opção 1 e determine um valor.");
                        break;
                    }
                    System.out.println("Escolha uma subpasta de descritores:");
                    File descriptorsBaseDir3 = new File("D:\\UFU\\tcc_video_frames\\descriptores");
                    File[] extractionSubdirs3 = descriptorsBaseDir3.listFiles(File::isDirectory);

                    if (extractionSubdirs3 != null && extractionSubdirs3.length > 0) {
                        for (int i = 0; i < extractionSubdirs3.length; i++) {
                            System.out.println((i + 1) + " - " + extractionSubdirs3[i].getName());
                        }
                        System.out.print("Escolha a subpasta de descritores para o agrupamento: ");
                        int dirChoice = scanner.nextInt();
                        scanner.nextLine();
                        if (dirChoice > 0 && dirChoice <= extractionSubdirs3.length) {
                            File chosenDescriptorSubdir = extractionSubdirs3[dirChoice - 1];
                            String[] possibleDescriptorFiles = {"descritoresHOG.txt", "descritoresORB.txt", "descritoresCNN.txt"};
                            boolean fileFound = false;
                            for (String descFileName : possibleDescriptorFiles) {
                                File descriptorFile = new File(chosenDescriptorSubdir, descFileName);
                                if (descriptorFile.exists()) {
                                    String chosenDescriptorsFilePath = descriptorFile.getAbsolutePath();
                                    try {
                                        frameService.sampleBySecond(chosenDescriptorsFilePath, samplingPorcentage);
                                        fileFound = true;
                                        break;
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            }
                            if (!fileFound) {
                                System.out.println("Nenhum arquivo de descritores válido foi encontrado na subpasta escolhida.");
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhuma subpasta de descritores encontrada.");
                    }
                    break;
                case 9:
                    System.out.println("Saindo...");
                    return;
                default:
                    System.out.println("Opção inválida.");
                    break;
            }
        }
    }
}
