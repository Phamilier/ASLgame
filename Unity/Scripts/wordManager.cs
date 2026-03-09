using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[System.Serializable]
public class SpellData
{
    public string word;
    public string element;
    public int power;

    public SpellData(string word, string element, int power)
    {
        this.word = word;
        this.element = element;
        this.power = power;
    }
}

public class wordManager : MonoBehaviour
{
    [SerializeField]
    public List<SpellData> spellDataList;
    private HashSet<string> wordSet;
    public List<string> GetWordStrings()
    {
        List<string> words = new List<string>();
        foreach (SpellData data in spellDataList)
        {
            words.Add(data.word);
        }
        return words;
    }

    // Start is called before the first frame update
    void Start()
    {
        GetWordStrings();
        //Debug.Log("Word Manager Initialized with " + spellDataList.Count + " spells.");
        //for (int i = 0; i < spellDataList.Count; i++)
        //{
        //    Debug.Log("Spell " + i + ": " + spellDataList[i].word + ", Element: " + spellDataList[i].element + ", Power: " + spellDataList[i].power);
        //}
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
