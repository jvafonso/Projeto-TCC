package com.projeto.tcc.services;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

@Service
public class MenuService {

    @Autowired
    private FrameService2 frameService;

    @Autowired
    private FrameExtractor frameExtractor;

    public void startMenu() {
        Scanner scanner = new Scanner(System.in);
        double similarityThreshold = 0;

        while (true) {
            System.out.println("\nEscolha uma opção:");
            System.out.println("1 - Definir taxa de similaridade");
            System.out.println("2 - Separar frames de um vídeo");
            System.out.println("3 - Extrair caracteristicas de um grupo de frames");
            System.out.println("4 - Agrupar frames");
            System.out.println("5 - Sair");
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
                    break;
                case 2:
                    System.out.print("Insira o caminho do arquivo de vídeo de entrada: ");
                    String videoPath = scanner.nextLine();
                    try {
                        frameService.samplingFrames(videoPath, similarityThreshold);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
                case 3:
                    if(similarityThreshold == 0) {
                        System.out.println("A taxa de similaridade não pode ser 0 escolha a opção 1 e determine um valor.");
                        break;
                    }
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
                                frameExtractor.extractFeatures(chosenFramesPath);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        } else {
                            System.out.println("Opção inválida.");
                        }
                    } else {
                        System.out.println("Nenhum conjunto de frames encontrado.");
                    }
                    break;
                case 4:
                    if (similarityThreshold == 0) {
                        System.out.println("A taxa de similaridade não pode ser 0 escolha a opção 1 e determine um valor.");
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
                        scanner.nextLine(); // Consume the newline character
                        if (dirChoice > 0 && dirChoice <= extractionSubdirs.length) {
                            File chosenDescriptorSubdir = extractionSubdirs[dirChoice - 1];
                            String[] possibleDescriptorFiles = {"descritoresHOG.txt", "descritoresORB.txt", "descritoresCNN.txt"};
                            boolean fileFound = false;
                            for (String descFileName : possibleDescriptorFiles) {
                                File descriptorFile = new File(chosenDescriptorSubdir, descFileName);
                                if (descriptorFile.exists()) {
                                    String chosenDescriptorsFilePath = descriptorFile.getAbsolutePath();
                                    try {
                                        frameService.groupFrames(chosenDescriptorsFilePath, similarityThreshold);
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
                    System.out.println("Saindo...");
                    return;
                default:
                    System.out.println("Opção inválida.");
                    break;
            }
        }
    }
}
