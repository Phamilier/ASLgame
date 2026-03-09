using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Globalization;
using System;
using System.IO;

public class ASLManager : MonoBehaviour
{
    public pyReceiver pyReceiver;
    public string firstLetter = "";
    public float firstConfidence = 0f;
    public string secondLetter = "";
    public float secondConfidence = 0f;
    public string currentWord = "";
    public wordManager wordManager;

    wordMatch matchResult;
    public GameObject fireball;
    public Camera mainCam;
    public float spawnDistance = 2f;
    public Vector3 spawnOffset = Vector3.zero; // optional fine-tune
    private bool hasSpawnedThisWord = false;
    private string lastDetectedLetter;
    private float holdTime = 0f;

    public struct wordMatch
    {
        public bool isMatch;
        public string word;
        public char nextLetter;
        public bool isComplete;
    }

    public wordMatch CheckMatch(string currentWord)
    {
        wordMatch result = new wordMatch();
        if (string.IsNullOrEmpty(currentWord)) return result;

        foreach(SpellData spellData in wordManager.spellDataList)
        {
            string word = spellData.word;
            if (word.StartsWith(currentWord, StringComparison.OrdinalIgnoreCase))
            {
                Debug.Log($"Current Word: {currentWord}, Matching Word: {word}");
                result.isMatch = true;
                result.word = word;
                if (word.Length > currentWord.Length)
                {
                    result.nextLetter = word[currentWord.Length];
                }
                else
                {
                    result.isComplete = true;
                }
                break;
            }
        }
        return result;
    }

    private void SpawnInFrontOfCamera(GameObject prefab)
    {
        if (!mainCam) mainCam = Camera.main;
        if (!prefab || !mainCam) return;

        Vector3 pos =
            mainCam.transform.position +
            mainCam.transform.forward * spawnDistance +
            mainCam.transform.TransformVector(spawnOffset);

        Quaternion rot = Quaternion.LookRotation(mainCam.transform.forward, mainCam.transform.up);

        Instantiate(prefab, pos, rot);
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        firstLetter = pyReceiver.firstLetter;
        secondLetter = pyReceiver.secondLetter;
        firstConfidence = pyReceiver.firstConfidence;
        secondConfidence = pyReceiver.secondConfidence;

        if (firstLetter == "V" && firstConfidence > 0.85f)
        {
            currentWord = "";
        }
        else if (firstConfidence > 0.85f && currentWord.Length == 0)
        {
            if(firstLetter != lastDetectedLetter)
            {
                lastDetectedLetter = firstLetter;
                holdTime = 0f;
            }
            else
            {
                holdTime += Time.deltaTime;
                if (holdTime > 0.5f)
                {
                    currentWord += firstLetter;
                    holdTime = 0f; // Reset hold time after adding letter
                }
                matchResult = CheckMatch(currentWord);
                if(!matchResult.isMatch) currentWord = "";
            }
        }

        if (currentWord.Length != 0 && firstLetter != lastDetectedLetter && secondLetter != lastDetectedLetter)
        {
            if (firstLetter == matchResult.nextLetter.ToString())
            {
                currentWord += firstLetter;
                lastDetectedLetter = firstLetter;
                matchResult = CheckMatch(currentWord);
            }
            else if (secondLetter == matchResult.nextLetter.ToString())
            {
                currentWord += secondLetter;
                lastDetectedLetter = secondLetter;
                matchResult = CheckMatch(currentWord);
            }
        }

        if (matchResult.isComplete)
        {
            if (!hasSpawnedThisWord && currentWord.Equals("FIRE", StringComparison.OrdinalIgnoreCase))
            {
                SpawnInFrontOfCamera(fireball);
                hasSpawnedThisWord = true;
            }

            // Reset so player can cast again
            //currentWord = "";
            //matchResult = new wordMatch();   // clears flags like isComplete
            //lastDetectedLetter = "";
            //holdTime = 0f;
        }
    }
}
